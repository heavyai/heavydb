#include "GroupByAndAggregate.h"
#include "AggregateUtils.h"

#include "ExpressionRange.h"
#include "InPlaceSort.h"
#include "GpuInitGroups.h"

#include "Execute.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "../CudaMgr/CudaMgr.h"
#include "../Shared/checked_alloc.h"
#include "../Utils/ChunkIter.h"
#include "DataMgr/BufferMgr/BufferMgr.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <numeric>
#include <thread>

#define AGGREGATE_ONE_VALUE(agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, sql_type__)                            \
  do {                                                                                                                 \
    if (sql_type__.is_fp()) {                                                                                          \
      if (chosen_bytes__ == sizeof(float)) {                                                                           \
        agg_##agg_kind__##_float(reinterpret_cast<int32_t*>(val_ptr__), *reinterpret_cast<const float*>(other_ptr__)); \
      } else {                                                                                                         \
        agg_##agg_kind__##_double(reinterpret_cast<int64_t*>(val_ptr__),                                               \
                                  *reinterpret_cast<const double*>(other_ptr__));                                      \
      }                                                                                                                \
    } else {                                                                                                           \
      if (chosen_bytes__ == sizeof(int32_t)) {                                                                         \
        agg_##agg_kind__##_int32(reinterpret_cast<int32_t*>(val_ptr__),                                                \
                                 *reinterpret_cast<const int32_t*>(other_ptr__));                                      \
      } else {                                                                                                         \
        agg_##agg_kind__(reinterpret_cast<int64_t*>(val_ptr__), *reinterpret_cast<const int64_t*>(other_ptr__));       \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)

#define AGGREGATE_ONE_NULLABLE_VALUE(agg_kind__, val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__)  \
  do {                                                                                                            \
    if (agg_info__.skip_null_val) {                                                                               \
      if (agg_info__.sql_type.is_fp()) {                                                                          \
        if (chosen_bytes__ == sizeof(float)) {                                                                    \
          agg_##agg_kind__##_float_skip_val(reinterpret_cast<int32_t*>(val_ptr__),                                \
                                            *reinterpret_cast<const float*>(other_ptr__),                         \
                                            *reinterpret_cast<const float*>(&init_val__));                        \
        } else {                                                                                                  \
          agg_##agg_kind__##_double_skip_val(reinterpret_cast<int64_t*>(val_ptr__),                               \
                                             *reinterpret_cast<const double*>(other_ptr__),                       \
                                             *reinterpret_cast<const double*>(&init_val__));                      \
        }                                                                                                         \
      } else {                                                                                                    \
        if (chosen_bytes__ == sizeof(int32_t)) {                                                                  \
          agg_##agg_kind__##_int32_skip_val(reinterpret_cast<int32_t*>(val_ptr__),                                \
                                            *reinterpret_cast<const int32_t*>(other_ptr__),                       \
                                            static_cast<int32_t>(init_val__));                                    \
        } else {                                                                                                  \
          agg_##agg_kind__##_skip_val(                                                                            \
              reinterpret_cast<int64_t*>(val_ptr__), *reinterpret_cast<const int64_t*>(other_ptr__), init_val__); \
        }                                                                                                         \
      }                                                                                                           \
    } else {                                                                                                      \
      AGGREGATE_ONE_VALUE(agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__.sql_type);               \
    }                                                                                                             \
  } while (0)

ResultRows::ResultRows(const QueryMemoryDescriptor& query_mem_desc,
                       const std::vector<Analyzer::Expr*>& targets,
                       const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                       int64_t* group_by_buffer,
                       const size_t groups_buffer_entry_count,
                       const bool output_columnar,
                       const std::vector<std::vector<const int8_t*>>& col_buffers,
                       const ExecutorDeviceType device_type,
                       const int device_id)
    : executor_(query_mem_desc.executor_),
      query_mem_desc_(query_mem_desc),
      row_set_mem_owner_(row_set_mem_owner),
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
  const bool is_group_by{!query_mem_desc.group_col_widths.empty()};
  for (const auto target_expr : targets) {
    const auto agg_info = target_info(target_expr);
    bool is_real_string = agg_info.sql_type.is_string() && agg_info.sql_type.get_compression() == kENCODING_NONE;
    bool is_array = agg_info.sql_type.is_array();
    CHECK(!is_real_string || !is_array);
    if (executor_->plan_state_->isLazyFetchColumn(target_expr) || is_real_string || is_array) {
      has_lazy_columns = true;
    }
    agg_args_.push_back(agg_arg_info(target_expr));
    targets_.push_back(agg_info);
  }
  std::vector<TargetValue> row;
  agg_init_vals_ =
      init_agg_val_vec(get_compact_targets(targets_, agg_args_), query_mem_desc.agg_col_widths.size(), is_group_by);
  if (in_place_ && has_lazy_columns) {
    while (fetchLazyOrBuildRow(row, col_buffers, targets, false, false, true)) {
    };
  }
  moveToBegin();
}

bool ResultRows::reduceSingleRow(const int8_t* row_ptr,
                                 const int32_t entry_count,
                                 const size_t bin,
                                 const int8_t warp_count,
                                 const bool is_columnar,
                                 const bool keep_cnt_dtnc_buff,
                                 std::vector<int64_t>& agg_vals) const {
  const size_t agg_col_count{agg_vals.size()};
  const auto row_size = query_mem_desc_.getRowSize();
  CHECK_EQ(agg_col_count, query_mem_desc_.agg_col_widths.size());
  CHECK_GE(agg_col_count, targets_.size());
  CHECK_EQ(is_columnar, query_mem_desc_.output_columnar);
  CHECK(query_mem_desc_.keyless_hash);
  const auto compact_targets = get_compact_targets(targets_, agg_args_);
  std::vector<int64_t> partial_agg_vals(agg_col_count, 0);
  bool discard_row = true;
  for (int8_t warp_idx = 0; warp_idx < warp_count; ++warp_idx) {
    bool discard_partial_result = true;
    for (size_t target_idx = 0, agg_col_idx = 0; target_idx < compact_targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      const auto& agg_info = compact_targets[target_idx];
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[agg_col_idx].compact;
      auto partial_bin_val = get_component(row_ptr + query_mem_desc_.getColOnlyOffInBytes(agg_col_idx), chosen_bytes);
      partial_agg_vals[agg_col_idx] = partial_bin_val;
      if (agg_info.is_distinct) {
        CHECK(agg_info.is_agg && agg_info.agg_kind == kCOUNT);
        partial_bin_val = bitmap_set_size(partial_bin_val, target_idx, row_set_mem_owner_->count_distinct_descriptors_);
        if (!keep_cnt_dtnc_buff)
          partial_agg_vals[target_idx] = partial_bin_val;
      }
      if (kAVG == agg_info.agg_kind) {
        CHECK(agg_info.is_agg && !agg_info.is_distinct);
        ++agg_col_idx;
        partial_bin_val = partial_agg_vals[agg_col_idx] =
            get_component(row_ptr + query_mem_desc_.getColOnlyOffInBytes(agg_col_idx),
                          query_mem_desc_.agg_col_widths[agg_col_idx].compact);
      }
      if (agg_col_idx == static_cast<size_t>(query_mem_desc_.idx_target_as_key) &&
          partial_bin_val != query_mem_desc_.init_val) {
        CHECK(agg_info.is_agg);
        discard_partial_result = false;
      }
    }
    row_ptr += row_size;
    if (discard_partial_result) {
      continue;
    }
    discard_row = false;
    for (size_t target_idx = 0, agg_col_idx = 0; target_idx < compact_targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      const auto& agg_info = compact_targets[target_idx];
      auto partial_bin_val = partial_agg_vals[agg_col_idx];
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[agg_col_idx].compact;
      if (agg_info.is_agg) {
        switch (agg_info.agg_kind) {
          case kAVG:
            AGGREGATE_ONE_VALUE(sum,
                                reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx + 1]),
                                reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx + 1]),
                                chosen_bytes,
                                agg_info.sql_type);
          // fall thru
          case kCOUNT:
          case kSUM:
            AGGREGATE_ONE_NULLABLE_VALUE(sum,
                                         reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                                         reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                                         agg_init_vals_[agg_col_idx],
                                         chosen_bytes,
                                         agg_info);
            break;
          case kMIN:
            AGGREGATE_ONE_NULLABLE_VALUE(min,
                                         reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                                         reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                                         agg_init_vals_[agg_col_idx],
                                         chosen_bytes,
                                         agg_info);
            break;
          case kMAX:
            AGGREGATE_ONE_NULLABLE_VALUE(max,
                                         reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                                         reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                                         agg_init_vals_[agg_col_idx],
                                         chosen_bytes,
                                         agg_info);
            break;
          default:
            CHECK(false);
            break;
        }
        if (agg_info.sql_type.is_integer() || agg_info.sql_type.is_decimal()) {
          switch (chosen_bytes) {
            case 8:
              break;
            case 4: {
              int32_t ret = *reinterpret_cast<const int32_t*>(&agg_vals[agg_col_idx]);
              agg_vals[agg_col_idx] = static_cast<int64_t>(ret);
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
  CHECK(!is_columnar || warp_count == 1);
  const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
  std::vector<int64_t> agg_vals(agg_col_count, 0);
  simple_keys_.reserve(groups_buffer_entry_count);
  target_values_.reserve(groups_buffer_entry_count);
  for (int32_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
    memcpy(&agg_vals[0], &agg_init_vals_[0], agg_col_count * sizeof(agg_vals[0]));
    beginRow(bin + min_val);
    if (reduceSingleRow(reinterpret_cast<const int8_t*>(group_by_buffer) + query_mem_desc_.getColOffInBytes(bin, 0),
                        groups_buffer_entry_count,
                        bin,
                        warp_count,
                        is_columnar,
                        true,
                        agg_vals)) {
      discardRow();
      continue;
    }
    addValues(agg_vals);
  }
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

void ResultRows::reduce(const ResultRows& other_results,
                        const QueryMemoryDescriptor& query_mem_desc,
                        const bool output_columnar) {
  if (other_results.definitelyHasNoRows()) {
    return;
  }
  if (definitelyHasNoRows()) {
    *this = other_results;
    return;
  }

  auto reduce_impl = [this](int8_t* crt_val_i1,
                            int8_t* crt_val_i2,
                            const int8_t* new_val_i1,
                            const int8_t* new_val_i2,
                            const TargetInfo& agg_info,
                            const int64_t agg_skip_val,
                            const size_t target_idx,
                            size_t crt_byte_width = sizeof(int64_t),
                            size_t next_byte_width = sizeof(int64_t)) {
    CHECK(agg_info.sql_type.is_integer() || agg_info.sql_type.is_decimal() || agg_info.sql_type.is_time() ||
          agg_info.sql_type.is_boolean() || agg_info.sql_type.is_string() || agg_info.sql_type.is_fp());
    switch (agg_info.agg_kind) {
      case kAVG:
        CHECK(crt_val_i2 && new_val_i2);
        AGGREGATE_ONE_VALUE(sum, crt_val_i2, new_val_i2, next_byte_width, agg_info.sql_type);
      // fall thru
      case kCOUNT:
        if (agg_info.is_distinct) {
          CHECK(agg_info.is_agg);
          CHECK_EQ(kCOUNT, agg_info.agg_kind);
          CHECK_EQ(crt_byte_width, sizeof(int64_t));
          auto crt_val_i1_ptr = reinterpret_cast<int64_t*>(crt_val_i1);
          auto count_distinct_desc_it = row_set_mem_owner_->count_distinct_descriptors_.find(target_idx);
          CHECK(count_distinct_desc_it != row_set_mem_owner_->count_distinct_descriptors_.end());
          auto old_set_ptr = reinterpret_cast<const int64_t*>(crt_val_i1_ptr);
          auto new_set_ptr = reinterpret_cast<const int64_t*>(new_val_i1);
          if (count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::Bitmap) {
            auto old_set = reinterpret_cast<int8_t*>(*old_set_ptr);
            auto new_set = reinterpret_cast<int8_t*>(*new_set_ptr);
            bitmap_set_unify(new_set, old_set, count_distinct_desc_it->second.bitmapSizeBytes());
          } else {
            CHECK(count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::StdSet);
            auto old_set = reinterpret_cast<std::set<int64_t>*>(*old_set_ptr);
            auto new_set = reinterpret_cast<std::set<int64_t>*>(*new_set_ptr);
            old_set->insert(new_set->begin(), new_set->end());
            new_set->insert(old_set->begin(), old_set->end());
          }
          break;
        }
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
  };

  const auto compact_targets = get_compact_targets(targets_, agg_args_);

  if (group_by_buffer_ && !in_place_) {
    CHECK(!query_mem_desc.sortOnGpu());
    CHECK(other_results.group_by_buffer_);
    CHECK(query_mem_desc.keyless_hash);
    CHECK(warp_count_ == 1 || !output_columnar);
    CHECK_EQ(static_cast<size_t>(warp_count_), query_mem_desc.getWarpCount());
    const size_t agg_col_count{query_mem_desc.agg_col_widths.size()};
    int8_t* crt_results = reinterpret_cast<int8_t*>(group_by_buffer_);
    int8_t* new_results = reinterpret_cast<int8_t*>(other_results.group_by_buffer_);
    size_t row_size = output_columnar ? 1 : query_mem_desc.getRowSize();
    CHECK_GE(agg_col_count, compact_targets.size());
    for (size_t target_index = 0, agg_col_idx = 0, col_base_off = query_mem_desc.getColOffInBytes(0, 0);
         target_index < compact_targets.size() && agg_col_idx < agg_col_count;
         ++target_index,
                col_base_off += query_mem_desc.getNextColOffInBytes(&crt_results[col_base_off], 0, agg_col_idx++)) {
      const auto agg_info = compact_targets[target_index];
      auto chosen_bytes = query_mem_desc.agg_col_widths[agg_col_idx].compact;
      auto next_chosen_bytes = chosen_bytes;
      if (kAVG == agg_info.agg_kind) {
        next_chosen_bytes = query_mem_desc.agg_col_widths[agg_col_idx + 1].compact;
      }

      for (size_t bin = 0, bin_base_off = col_base_off; bin < groups_buffer_entry_count_;
           ++bin, bin_base_off += query_mem_desc.getColOffInBytesInNextBin(agg_col_idx)) {
        int8_t* crt_next_result_ptr{nullptr};
        int8_t* new_next_result_ptr{nullptr};
        if (kAVG == agg_info.agg_kind) {
          auto additional_off =
              bin_base_off + query_mem_desc.getNextColOffInBytes(&crt_results[bin_base_off], bin, agg_col_idx);
          crt_next_result_ptr = &crt_results[additional_off];
          new_next_result_ptr = &new_results[additional_off];
        }

        for (size_t warp_idx = 0, row_base_off = bin_base_off; warp_idx < static_cast<size_t>(warp_count_);
             ++warp_idx, row_base_off += row_size) {
          reduce_impl(&crt_results[row_base_off],
                      crt_next_result_ptr,
                      &new_results[row_base_off],
                      new_next_result_ptr,
                      agg_info,
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
        col_base_off += query_mem_desc.getNextColOffInBytes(&crt_results[col_base_off], 0, agg_col_idx++);
      }
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
      reduce_impl(reinterpret_cast<int8_t*>(&crt_results[agg_col_idx].i1),
                  reinterpret_cast<int8_t*>(&crt_results[agg_col_idx].i2),
                  reinterpret_cast<const int8_t*>(&new_results[agg_col_idx].i1),
                  reinterpret_cast<const int8_t*>(&new_results[agg_col_idx].i2),
                  compact_targets[agg_col_idx],
                  get_initial_val(compact_targets[agg_col_idx]),
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

    std::function<void(const bool output_columnar,
                       int32_t& groups_buffer_entry_count,
                       const int32_t other_groups_buffer_entry_count,
                       int64_t** group_by_buffer_ptr,
                       const int64_t* other_group_by_buffer,
                       const GroupByColRangeType hash_type,
                       const std::vector<TargetInfo>& targets,
                       const QueryMemoryDescriptor& query_mem_desc_in,
                       std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)> reduce_in_place;

    reduce_in_place = [this, &reduce_impl, &reduce_in_place](const bool output_columnar,
                                                             int32_t& groups_buffer_entry_count,
                                                             const int32_t other_groups_buffer_entry_count,
                                                             int64_t** group_by_buffer_ptr,
                                                             const int64_t* other_group_by_buffer,
                                                             const GroupByColRangeType hash_type,
                                                             const std::vector<TargetInfo>& targets,
                                                             const QueryMemoryDescriptor& query_mem_desc_in,
                                                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
      const size_t group_by_col_count{query_mem_desc_in.group_col_widths.size()};
      const int64_t min_val{query_mem_desc_in.min_val};
      const size_t row_size_quad{output_columnar ? 0 : query_mem_desc_in.getRowSize() / sizeof(int64_t)};
      CHECK_LE(other_groups_buffer_entry_count, groups_buffer_entry_count);
      CHECK_EQ(query_mem_desc_in.output_columnar, output_columnar);
      for (size_t bin = 0, bin_base_off = query_mem_desc_in.getColOffInBytes(0, 0);
           bin < static_cast<size_t>(other_groups_buffer_entry_count);
           ++bin, bin_base_off += query_mem_desc_in.getColOffInBytesInNextBin(0)) {
        const size_t other_key_off = query_mem_desc_in.getKeyOffInBytes(bin) / sizeof(int64_t);
        const auto other_key_buff = &other_group_by_buffer[other_key_off];
        if (other_key_buff[0] == EMPTY_KEY_64) {
          continue;
        }
        int64_t* group_val_buff{nullptr};
        size_t target_bin{0};
        switch (hash_type) {
          case GroupByColRangeType::OneColKnownRange:
            if (output_columnar) {
              target_bin = get_columnar_group_bin_offset(
                  *group_by_buffer_ptr, other_key_buff[0], min_val, query_mem_desc_in.bucket);
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
            group_val_buff = get_group_value(
                *group_by_buffer_ptr, groups_buffer_entry_count, other_key_buff, group_by_col_count, row_size_quad);
            break;
          default:
            CHECK(false);
        }
        if (!output_columnar && !group_val_buff) {
          throw ReductionRanOutOfSlots();
        }
        if (hash_type == GroupByColRangeType::OneColKnownRange &&
            other_group_by_buffer[other_key_off] != (*group_by_buffer_ptr)[other_key_off]) {
          CHECK(EMPTY_KEY_64 == other_group_by_buffer[other_key_off] ||
                EMPTY_KEY_64 == (*group_by_buffer_ptr)[other_key_off]);
        }
        size_t target_idx{0};
        size_t col_idx{0};
        size_t other_off{bin_base_off};
        for (const auto& agg_info : targets) {
          const auto chosen_bytes = query_mem_desc_in.agg_col_widths[col_idx].compact;
          auto next_chosen_bytes = chosen_bytes;
          const auto other_ptr = reinterpret_cast<const int8_t*>(other_group_by_buffer) + other_off;
          const auto other_val = get_component(other_ptr, chosen_bytes);
          int8_t* col_ptr{nullptr};
          if (output_columnar) {
            col_ptr = reinterpret_cast<int8_t*>(*group_by_buffer_ptr) +
                      query_mem_desc_in.getColOffInBytes(target_bin, col_idx);
          } else {
            col_ptr = reinterpret_cast<int8_t*>(group_val_buff) + query_mem_desc_in.getColOnlyOffInBytes(col_idx);
          }

          int8_t* next_col_ptr{nullptr};
          const int8_t* other_next_ptr{nullptr};
          if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
            if (output_columnar) {
              next_col_ptr = reinterpret_cast<int8_t*>(*group_by_buffer_ptr) +
                             query_mem_desc_in.getColOffInBytes(target_bin, col_idx + 1);
            } else {
              next_col_ptr =
                  reinterpret_cast<int8_t*>(group_val_buff) + query_mem_desc_in.getColOnlyOffInBytes(col_idx + 1);
            }
            next_chosen_bytes = query_mem_desc_in.agg_col_widths[col_idx + 1].compact;
            other_off += query_mem_desc_in.getNextColOffInBytes(other_ptr, bin, col_idx + 1);
            other_next_ptr = reinterpret_cast<const int8_t*>(other_group_by_buffer) + other_off;
            ++col_idx;
          }

          switch (chosen_bytes) {
            case 4: {
              if (agg_info.is_agg) {
                reduce_impl(col_ptr,
                            next_col_ptr,
                            other_ptr,
                            other_next_ptr,
                            agg_info,
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
                reduce_impl(col_ptr,
                            next_col_ptr,
                            other_ptr,
                            other_next_ptr,
                            agg_info,
                            agg_init_vals_[col_idx],
                            target_idx,
                            chosen_bytes,
                            next_chosen_bytes);
              } else {
                *reinterpret_cast<int64_t*>(col_ptr) = other_val;
              }
              break;
            }
            case 1:
            case 2:
            default:
              CHECK(false);
          }
          other_off += query_mem_desc_in.getNextColOffInBytes(other_ptr, bin, col_idx);
          ++col_idx;
          ++target_idx;
        }
      }
    };

    if (query_mem_desc.getSmallBufferSizeBytes()) {
      CHECK_EQ(size_t(2), in_place_groups_by_buffers_entry_count_.size());
      CHECK_EQ(size_t(2), in_place_group_by_buffers_.size());
      CHECK(!output_columnar_);
      reduce_in_place(output_columnar_,
                      in_place_groups_by_buffers_entry_count_[0],
                      other_results.in_place_groups_by_buffers_entry_count_[0],
                      group_by_buffer_ptr,
                      other_group_by_buffer,
                      GroupByColRangeType::OneColKnownRange,
                      compact_targets,
                      query_mem_desc,
                      row_set_mem_owner_);
      group_by_buffer_ptr = &in_place_group_by_buffers_[1];
      other_group_by_buffer = other_results.in_place_group_by_buffers_[1];
      reduce_in_place(output_columnar_,
                      in_place_groups_by_buffers_entry_count_[1],
                      other_results.in_place_groups_by_buffers_entry_count_[1],
                      group_by_buffer_ptr,
                      other_group_by_buffer,
                      GroupByColRangeType::MultiCol,
                      compact_targets,
                      query_mem_desc,
                      row_set_mem_owner_);
    } else {
      reduce_in_place(output_columnar_,
                      in_place_groups_by_buffers_entry_count_[0],
                      other_results.in_place_groups_by_buffers_entry_count_[0],
                      group_by_buffer_ptr,
                      other_group_by_buffer,
                      query_mem_desc.hash_type,
                      compact_targets,
                      query_mem_desc,
                      row_set_mem_owner_);
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
      reduce_impl(reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i1),
                  reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i2),
                  reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i1),
                  reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i2),
                  compact_targets[agg_col_idx],
                  get_initial_val(compact_targets[agg_col_idx]),
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
      const auto agg_info = compact_targets[agg_col_idx];
      if (agg_info.is_agg) {
        reduce_impl(reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i1),
                    reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i2),
                    reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i1),
                    reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i2),
                    agg_info,
                    get_initial_val(agg_info),
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

namespace {

__attribute__((always_inline)) inline double pair_to_double(const std::pair<int64_t, int64_t>& fp_pair,
                                                            const SQLTypeInfo& ti) {
  double dividend{0.0};
  int64_t null_val{0};
  switch (ti.get_type()) {
    case kFLOAT: {
      dividend = *reinterpret_cast<const double*>(&fp_pair.first);
      double null_float = inline_fp_null_val(ti);
      null_val = *reinterpret_cast<int64_t*>(&null_float);
      break;
    }
    case kDOUBLE: {
      dividend = *reinterpret_cast<const double*>(&fp_pair.first);
      double null_double = inline_fp_null_val(ti);
      null_val = *reinterpret_cast<int64_t*>(&null_double);
      break;
    }
    default: {
      CHECK(ti.is_integer() || ti.is_decimal());
      dividend = static_cast<double>(fp_pair.first);
      null_val = inline_int_null_val(ti);
      break;
    }
  }
  if (!ti.get_notnull() && null_val == fp_pair.first) {
    return inline_fp_null_val(SQLTypeInfo(kDOUBLE, false));
  }

  return ti.is_integer() || ti.is_decimal()
             ? (dividend / exp_to_scale(ti.is_decimal() ? ti.get_scale() : 0)) / static_cast<double>(fp_pair.second)
             : dividend / static_cast<double>(fp_pair.second);
}

}  // namespace

void ResultRows::sort(const std::list<Analyzer::OrderEntry>& order_entries,
                      const bool remove_duplicates,
                      const int64_t top_n) {
  if (definitelyHasNoRows()) {
    return;
  }
  if (query_mem_desc_.sortOnGpu()) {
    try {
      inplaceSortGpu(order_entries);
    } catch (const OutOfMemory&) {
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
  const auto compact_targets = get_compact_targets(targets_, agg_args_);
  const bool use_heap{order_entries.size() == 1 && !remove_duplicates && top_n};
  auto compare = [this, &order_entries, compact_targets, use_heap](const InternalRow& lhs, const InternalRow& rhs) {
    // NB: The compare function must define a strict weak ordering, otherwise
    // std::sort will trigger a segmentation fault (or corrupt memory).
    for (const auto order_entry : order_entries) {
      CHECK_GE(order_entry.tle_no, 1);
      const auto& entry_ti = compact_targets[order_entry.tle_no - 1].sql_type;
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
          CHECK_EQ(4, entry_ti.get_size());
          auto string_dict = executor_->getStringDictionary(entry_ti.get_comp_param(), row_set_mem_owner_);
          auto lhs_str = string_dict->getString(lhs_v.i1);
          auto rhs_str = string_dict->getString(rhs_v.i1);
          if (lhs_str == rhs_str) {
            continue;
          }
          return use_desc_cmp ? lhs_str > rhs_str : lhs_str < rhs_str;
        }
        if (UNLIKELY(compact_targets[order_entry.tle_no - 1].is_distinct)) {
          const auto lhs_sz =
              bitmap_set_size(lhs_v.i1, order_entry.tle_no - 1, row_set_mem_owner_->count_distinct_descriptors_);
          const auto rhs_sz =
              bitmap_set_size(rhs_v.i1, order_entry.tle_no - 1, row_set_mem_owner_->count_distinct_descriptors_);
          if (lhs_sz == rhs_sz) {
            continue;
          }
          return use_desc_cmp ? lhs_sz > rhs_sz : lhs_sz < rhs_sz;
        }
        if (lhs_v.i1 == rhs_v.i1) {
          continue;
        }
        return use_desc_cmp ? lhs_v.i1 > rhs_v.i1 : lhs_v.i1 < rhs_v.i1;
      } else {
        if (lhs_v.isPair()) {
          CHECK(rhs_v.isPair());
          const auto lhs = pair_to_double({lhs_v.i1, lhs_v.i2}, entry_ti);
          const auto rhs = pair_to_double({rhs_v.i1, rhs_v.i2}, entry_ti);
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
  ScopedScratchBuffer scratch_buff(query_mem_desc_.entry_count * sizeof(int64_t), data_mgr, device_id);
  auto tmp_buff = reinterpret_cast<int64_t*>(scratch_buff.getPtr());
  CHECK_EQ(size_t(1), order_entries.size());
  const auto idx_buff = gpu_query_mem.group_by_buffers.second - query_mem_desc_.entry_count * sizeof(int64_t);
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc_.getColOffInBytes(0, target_idx);
    const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
    sort_groups_gpu(reinterpret_cast<int64_t*>(val_buff),
                    reinterpret_cast<int64_t*>(idx_buff),
                    query_mem_desc_.entry_count,
                    order_entry.is_desc,
                    chosen_bytes);
    if (!query_mem_desc_.keyless_hash) {
      apply_permutation_gpu(reinterpret_cast<int64_t*>(gpu_query_mem.group_by_buffers.second),
                            reinterpret_cast<int64_t*>(idx_buff),
                            query_mem_desc_.entry_count,
                            tmp_buff,
                            sizeof(int64_t));
    }
    for (size_t target_idx = 0; target_idx < query_mem_desc_.agg_col_widths.size(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
      const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc_.getColOffInBytes(0, target_idx);
      apply_permutation_gpu(reinterpret_cast<int64_t*>(val_buff),
                            reinterpret_cast<int64_t*>(idx_buff),
                            query_mem_desc_.entry_count,
                            tmp_buff,
                            chosen_bytes);
    }
  }
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

void ResultRows::inplaceSortCpu(const std::list<Analyzer::OrderEntry>& order_entries) {
  CHECK(in_place_);
  CHECK(!query_mem_desc_.keyless_hash);
  CHECK_EQ(size_t(1), in_place_group_by_buffers_.size());
  std::vector<int64_t> tmp_buff(query_mem_desc_.entry_count);
  std::vector<int64_t> idx_buff(query_mem_desc_.entry_count);
  CHECK_EQ(size_t(1), order_entries.size());
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto sortkey_val_buff = in_place_group_by_buffers_.front() + order_entry.tle_no * query_mem_desc_.entry_count;
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
          in_place_group_by_buffers_.front() + (target_idx + 1) * query_mem_desc_.entry_count;
      apply_permutation_cpu(satellite_val_buff, &idx_buff[0], query_mem_desc_.entry_count, &tmp_buff[0], chosen_bytes);
    }
  }
}

#undef UNLIKELY
#undef LIKELY

namespace {

TargetValue result_rows_get_impl(const InternalTargetValue& col_val,
                                 const int64_t agg_initial_val,
                                 const size_t col_idx,
                                 const TargetInfo& agg_info,
                                 const SQLTypeInfo& target_type,
                                 const bool decimal_to_double,
                                 const bool translate_strings,
                                 const Executor* executor,
                                 const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  const auto& ti = agg_info.sql_type;
  if (agg_info.agg_kind == kAVG) {
    CHECK(!ti.is_string());
    CHECK(col_val.isPair());
    return pair_to_double({col_val.i1, col_val.i2}, ti);
  }
  if (ti.is_integer() || ti.is_decimal() || ti.is_boolean() || ti.is_time()) {
    if (agg_info.is_distinct) {
      return TargetValue(bitmap_set_size(col_val.i1, col_idx, row_set_mem_owner->getCountDistinctDescriptors()));
    }
    CHECK(col_val.isInt());
    if (ti.is_decimal() && decimal_to_double) {
      if (col_val.i1 == inline_int_null_val(SQLTypeInfo(decimal_to_int_type(ti), false))) {
        return NULL_DOUBLE;
      }
      return static_cast<double>(col_val.i1) / exp_to_scale(ti.get_scale());
    }
    if (inline_int_null_val(ti) == col_val.i1) {
      return inline_int_null_val(target_type);
    }
    return col_val.i1;
  } else if (ti.is_string()) {
    if (ti.get_compression() == kENCODING_DICT) {
      const int dict_id = ti.get_comp_param();
      const auto string_id = col_val.i1;
      if (!translate_strings) {
        return TargetValue(string_id);
      }
      return string_id == NULL_INT
                 ? TargetValue(nullptr)
                 : TargetValue(executor->getStringDictionary(dict_id, row_set_mem_owner)->getString(string_id));
    } else {
      CHECK_EQ(kENCODING_NONE, ti.get_compression());
      return col_val.isNull() ? TargetValue(nullptr) : TargetValue(col_val.strVal());
    }
  } else if (ti.is_array()) {
    const auto& elem_type = ti.get_elem_type();
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
          tv_arr.emplace_back(*reinterpret_cast<const double*>(&x));
        }
      }
    } else if (elem_type.is_string()) {
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
      const auto& string_ids = *reinterpret_cast<std::vector<int64_t>*>(col_val.i1);
      const int dict_id = ti.get_comp_param();
      for (const auto string_id : string_ids) {
        tv_arr.emplace_back(
            string_id == NULL_INT
                ? NullableString(nullptr)
                : NullableString(executor->getStringDictionary(dict_id, row_set_mem_owner)->getString(string_id)));
      }
    } else {
      CHECK(false);
    }
    return tv_arr;
  } else {
    CHECK(ti.is_fp());
    if (ti.get_type() == kFLOAT) {
      return ScalarTargetValue(static_cast<float>(*reinterpret_cast<const double*>(&col_val.i1)));
    }
    return ScalarTargetValue(*reinterpret_cast<const double*>(&col_val.i1));
  }
  CHECK(false);
}

int64_t lazy_decode(const Analyzer::ColumnVar* col_var, const int8_t* byte_stream, const int64_t pos) {
  const auto enc_type = col_var->get_compression();
  const auto& type_info = col_var->get_type_info();
  if (type_info.is_fp()) {
    if (type_info.get_type() == kFLOAT) {
      float fval = fixed_width_float_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<int32_t*>(&fval);
    } else {
      double fval = fixed_width_double_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<int64_t*>(&fval);
    }
  }
  CHECK(type_info.is_integer() || type_info.is_decimal() || type_info.is_time() || type_info.is_boolean() ||
        (type_info.is_string() && enc_type == kENCODING_DICT));
  size_t type_bitwidth = get_bit_width(col_var->get_type_info());
  if (col_var->get_type_info().get_compression() == kENCODING_FIXED) {
    type_bitwidth = col_var->get_type_info().get_comp_param();
  }
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  return fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
}

template <class T>
int64_t arr_elem_bitcast(const T val) {
  return val;
}

template <>
int64_t arr_elem_bitcast(const float val) {
  const double dval{val};
  return *reinterpret_cast<const int64_t*>(&dval);
}

template <>
int64_t arr_elem_bitcast(const double val) {
  return *reinterpret_cast<const int64_t*>(&val);
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

TargetValue ResultRows::getRowAt(const size_t row_idx,
                                 const size_t col_idx,
                                 const bool translate_strings,
                                 const bool decimal_to_double /* = true */) const {
  if (just_explain_) {
    return explanation_;
  }

  if (in_place_ || group_by_buffer_) {
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
                              agg_init_vals_[agg_col_idx],
                              col_idx,
                              get_compact_target(targets_[col_idx], agg_args_[col_idx]),
                              targets_[col_idx].sql_type,
                              decimal_to_double,
                              translate_strings,
                              executor_,
                              row_set_mem_owner_);
}

std::vector<TargetValue> ResultRows::getNextRow(const bool translate_strings, const bool decimal_to_double) const {
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

bool ResultRows::fetchLazyOrBuildRow(std::vector<TargetValue>& row,
                                     const std::vector<std::vector<const int8_t*>>& col_buffers,
                                     const std::vector<Analyzer::Expr*>& targets,
                                     const bool translate_strings,
                                     const bool decimal_to_double,
                                     const bool fetch_lazy) const {
  const auto compact_targets = get_compact_targets(targets_, agg_args_);
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
                          groups_buffer_entry_count_,
                          group_by_buffer_idx_,
                          warp_count,
                          output_columnar_,
                          false,
                          agg_vals)) {
        continue;
      }
      for (size_t target_idx = 0, agg_col_idx = 0; target_idx < compact_targets.size() && agg_col_idx < agg_col_count;
           ++target_idx, ++agg_col_idx) {
        const auto& agg_info = compact_targets[target_idx];
        if (agg_info.is_distinct) {
          row.emplace_back(agg_vals[agg_col_idx]);
        } else {
          const auto chosen_bytes = query_mem_desc_.agg_col_widths[agg_col_idx].compact;
          if (compact_targets[target_idx].sql_type.is_fp() && chosen_bytes == sizeof(float)) {
            agg_vals[agg_col_idx] = float_to_double_bin(agg_vals[agg_col_idx], !agg_info.sql_type.get_notnull());
          }
          auto target_val =
              (kAVG == agg_info.agg_kind ? InternalTargetValue(agg_vals[agg_col_idx], agg_vals[agg_col_idx + 1])
                                         : InternalTargetValue(agg_vals[agg_col_idx]));
          row.push_back(result_rows_get_impl(target_val,
                                             agg_init_vals_[agg_col_idx],
                                             target_idx,
                                             compact_targets[target_idx],
                                             targets_[target_idx].sql_type,
                                             decimal_to_double,
                                             translate_strings,
                                             executor_,
                                             row_set_mem_owner_));
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
        auto chosen_bytes = query_mem_desc_.agg_col_widths[out_vec_idx].compact;
        size_t next_chosen_bytes = chosen_bytes;
        auto val1 = get_component(col_ptr, chosen_bytes);
        const auto& agg_info = compact_targets[col_idx];
        if (agg_info.sql_type.is_fp() && chosen_bytes == sizeof(float)) {
          val1 = float_to_double_bin(val1, !agg_info.sql_type.get_notnull());
        }
        bool is_real_string = agg_info.sql_type.is_string() && agg_info.sql_type.get_compression() == kENCODING_NONE;
        bool is_array = agg_info.sql_type.is_array();
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
                switch (elem_ti.get_size()) {
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
              elem_sz = elem_ti.get_size();
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
          if (next_col_ptr) {
            col_ptr = next_col_ptr;
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
          row.push_back(result_rows_get_impl(
              build_itv(val1, val2),
              (agg_info.agg_kind == kAVG ? agg_init_vals_[out_vec_idx - 1] : agg_init_vals_[out_vec_idx]),
              col_idx,
              agg_info,
              targets_[col_idx].sql_type,
              decimal_to_double,
              translate_strings,
              executor_,
              row_set_mem_owner_));
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
    return val.i1 == *reinterpret_cast<const int64_t*>(&null_val);
  }
  if (val.isPair()) {
    return val.i2 == 0;
  }
  if (val.isStr()) {
    return false;
  }
  CHECK(val.isNull());
  return true;
}

QueryExecutionContext::QueryExecutionContext(const QueryMemoryDescriptor& query_mem_desc,
                                             const std::vector<int64_t>& init_agg_vals,
                                             const Executor* executor,
                                             const ExecutorDeviceType device_type,
                                             const int device_id,
                                             const std::vector<std::vector<const int8_t*>>& col_buffers,
                                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                             const bool output_columnar,
                                             const bool sort_on_gpu,
                                             RenderAllocatorMap* render_allocator_map)
    : query_mem_desc_(query_mem_desc),
      init_agg_vals_(executor->plan_state_->init_agg_vals_),
      executor_(executor),
      device_type_(device_type),
      device_id_(device_id),
      col_buffers_(col_buffers),
      num_buffers_{device_type == ExecutorDeviceType::CPU
                       ? 1
                       : executor->blockSize() * (query_mem_desc_.blocksShareMemory() ? 1 : executor->gridSize())},
      row_set_mem_owner_(row_set_mem_owner),
      output_columnar_(output_columnar),
      sort_on_gpu_(sort_on_gpu) {
  CHECK(!sort_on_gpu_ || output_columnar);
  if (render_allocator_map || query_mem_desc_.group_col_widths.empty()) {
    allocateCountDistinctBuffers(false);
    return;
  }

  std::vector<int64_t> group_by_buffer_template(query_mem_desc_.getBufferSizeQuad(device_type));
  if (!query_mem_desc_.lazyInitGroups(device_type)) {
    if (output_columnar_) {
      initColumnarGroups(
          &group_by_buffer_template[0], &init_agg_vals[0], query_mem_desc_.entry_count, query_mem_desc_.keyless_hash);
    } else {
      initGroups(&group_by_buffer_template[0],
                 &init_agg_vals[0],
                 query_mem_desc_.entry_count,
                 query_mem_desc_.keyless_hash,
                 query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1);
    }
  }

  if (query_mem_desc_.interleavedBins(device_type_)) {
    CHECK(query_mem_desc_.keyless_hash);
  }

  if (query_mem_desc_.keyless_hash) {
    CHECK_EQ(size_t(0), query_mem_desc_.getSmallBufferSizeQuad());
  }

  std::vector<int64_t> group_by_small_buffer_template;
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    CHECK(!output_columnar_ && !query_mem_desc_.keyless_hash);
    group_by_small_buffer_template.resize(query_mem_desc_.getSmallBufferSizeQuad());
    initGroups(&group_by_small_buffer_template[0], &init_agg_vals[0], query_mem_desc_.entry_count_small, false, 1);
  }

  size_t step{device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory() ? executor_->blockSize()
                                                                                              : 1};

  for (size_t i = 0; i < num_buffers_; i += step) {
    size_t index_buffer_qw{device_type_ == ExecutorDeviceType::GPU && sort_on_gpu_ && query_mem_desc_.keyless_hash
                               ? query_mem_desc_.entry_count
                               : 0};
    auto group_by_buffer = static_cast<int64_t*>(
        checked_malloc(query_mem_desc_.getBufferSizeBytes(device_type_) + index_buffer_qw * sizeof(int64_t)));
    if (!query_mem_desc_.lazyInitGroups(device_type)) {
      memcpy(group_by_buffer + index_buffer_qw,
             &group_by_buffer_template[0],
             query_mem_desc_.getBufferSizeBytes(device_type_));
    }
    row_set_mem_owner_->addGroupByBuffer(group_by_buffer);
    group_by_buffers_.push_back(group_by_buffer);
    for (size_t j = 1; j < step; ++j) {
      group_by_buffers_.push_back(nullptr);
    }
    if (query_mem_desc_.getSmallBufferSizeBytes()) {
      auto group_by_small_buffer = static_cast<int64_t*>(checked_malloc(query_mem_desc_.getSmallBufferSizeBytes()));
      row_set_mem_owner_->addGroupByBuffer(group_by_small_buffer);
      memcpy(group_by_small_buffer, &group_by_small_buffer_template[0], query_mem_desc_.getSmallBufferSizeBytes());
      small_group_by_buffers_.push_back(group_by_small_buffer);
      for (size_t j = 1; j < step; ++j) {
        small_group_by_buffers_.push_back(nullptr);
      }
    }
  }
}

void QueryExecutionContext::initColumnPerRow(int8_t* row_ptr,
                                             const size_t bin,
                                             const int64_t* init_vals,
                                             const std::vector<ssize_t>& bitmap_sizes) {
  int8_t* col_ptr = row_ptr;
  for (size_t col_idx = 0; col_idx < query_mem_desc_.agg_col_widths.size();
       col_ptr += query_mem_desc_.getNextColOffInBytes(col_ptr, bin, col_idx++)) {
    const ssize_t bm_sz{bitmap_sizes[col_idx]};
    int64_t init_val{0};
    if (!bm_sz) {
      init_val = init_vals[col_idx];
    } else {
      CHECK_EQ(static_cast<size_t>(query_mem_desc_.agg_col_widths[col_idx].compact), sizeof(int64_t));
      init_val = bm_sz > 0 ? allocateCountDistinctBitmap(bm_sz) : allocateCountDistinctSet();
    }
    switch (query_mem_desc_.agg_col_widths[col_idx].compact) {
      case 1:
        *col_ptr = static_cast<int8_t>(init_val);
        break;
      case 2:
        *reinterpret_cast<int16_t*>(col_ptr) = (int16_t)init_val;
        break;
      case 4:
        *reinterpret_cast<int32_t*>(col_ptr) = (int32_t)init_val;
        break;
      case 8:
        *reinterpret_cast<int64_t*>(col_ptr) = init_val;
        break;
      default:
        CHECK(false);
    }
  }
}

void QueryExecutionContext::initGroups(int64_t* groups_buffer,
                                       const int64_t* init_vals,
                                       const int32_t groups_buffer_entry_count,
                                       const bool keyless,
                                       const size_t warp_size) {
  const size_t key_qw_count{query_mem_desc_.group_col_widths.size()};
  const size_t row_size{query_mem_desc_.getRowSize()};
  const size_t col_base_off{query_mem_desc_.getColOffInBytes(0, 0)};

  auto agg_bitmap_size = allocateCountDistinctBuffers(true);
  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);

  if (keyless) {
    CHECK(warp_size >= 1);
    CHECK(key_qw_count == 1);
    for (size_t warp_idx = 0; warp_idx < warp_size; ++warp_idx) {
      for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count); ++bin, buffer_ptr += row_size) {
        initColumnPerRow(&buffer_ptr[col_base_off], bin, init_vals, agg_bitmap_size);
      }
    }
    return;
  }

  for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count); ++bin, buffer_ptr += row_size) {
    for (size_t key_idx = 0; key_idx < key_qw_count; ++key_idx) {
      reinterpret_cast<int64_t*>(buffer_ptr)[key_idx] = EMPTY_KEY_64;
    }
    initColumnPerRow(&buffer_ptr[col_base_off], bin, init_vals, agg_bitmap_size);
  }
}

template <typename T>
int8_t* QueryExecutionContext::initColumnarBuffer(T* buffer_ptr,
                                                  const T init_val,
                                                  const uint32_t entry_count,
                                                  const ssize_t bitmap_sz,
                                                  const bool key_or_col) {
  static_assert(sizeof(T) <= sizeof(int64_t), "Unsupported template type");
  if (key_or_col) {
    for (uint32_t i = 0; i < entry_count; ++i) {
      buffer_ptr[i] = init_val;
    }
  } else {
    for (uint32_t j = 0; j < entry_count; ++j) {
      if (!bitmap_sz) {
        buffer_ptr[j] = init_val;
      } else {
        CHECK_EQ(sizeof(int64_t), sizeof(T));
        buffer_ptr[j] = bitmap_sz > 0 ? allocateCountDistinctBitmap(bitmap_sz) : allocateCountDistinctSet();
      }
    }
  }

  return reinterpret_cast<int8_t*>(align_to_int64(buffer_ptr + entry_count));
}

void QueryExecutionContext::initColumnarGroups(int64_t* groups_buffer,
                                               const int64_t* init_vals,
                                               const int32_t groups_buffer_entry_count,
                                               const bool keyless) {
  auto agg_bitmap_size = allocateCountDistinctBuffers(true);
  const int32_t agg_col_count = query_mem_desc_.agg_col_widths.size();
  const int32_t key_qw_count = query_mem_desc_.group_col_widths.size();
  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);
  CHECK(key_qw_count == 1);
  if (!keyless) {
    buffer_ptr =
        initColumnarBuffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr), EMPTY_KEY_64, groups_buffer_entry_count);
  }
  for (int32_t i = 0; i < agg_col_count; ++i) {
    const ssize_t bitmap_sz{agg_bitmap_size[i]};
    switch (query_mem_desc_.agg_col_widths[i].compact) {
      case 1:
        buffer_ptr = initColumnarBuffer<int8_t>(buffer_ptr, init_vals[i], bitmap_sz, false);
        break;
      case 2:
        buffer_ptr =
            initColumnarBuffer<int16_t>(reinterpret_cast<int16_t*>(buffer_ptr), init_vals[i], bitmap_sz, false);
        break;
      case 4:
        buffer_ptr =
            initColumnarBuffer<int32_t>(reinterpret_cast<int32_t*>(buffer_ptr), init_vals[i], bitmap_sz, false);
        break;
      case 8:
        buffer_ptr =
            initColumnarBuffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr), init_vals[i], bitmap_sz, false);
        break;
      default:
        CHECK(false);
    }
  }
}

// deferred is true for group by queries; initGroups will allocate a bitmap
// for each group slot
std::vector<ssize_t> QueryExecutionContext::allocateCountDistinctBuffers(const bool deferred) {
  const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
  std::vector<ssize_t> agg_bitmap_size(deferred ? agg_col_count : 0);

  CHECK_GE(agg_col_count, executor_->plan_state_->target_exprs_.size());
  for (size_t target_idx = 0, agg_col_idx = 0;
       target_idx < executor_->plan_state_->target_exprs_.size() && agg_col_idx < agg_col_count;
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = executor_->plan_state_->target_exprs_[target_idx];
    const auto agg_info = target_info(target_expr);
    if (agg_info.is_distinct) {
      CHECK(agg_info.is_agg && agg_info.agg_kind == kCOUNT);
      CHECK_EQ(static_cast<size_t>(query_mem_desc_.agg_col_widths[agg_col_idx].actual), sizeof(int64_t));
      auto count_distinct_it = query_mem_desc_.count_distinct_descriptors_.find(target_idx);
      CHECK(count_distinct_it != query_mem_desc_.count_distinct_descriptors_.end());
      const auto& count_distinct_desc = count_distinct_it->second;
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        if (deferred) {
          agg_bitmap_size[agg_col_idx] = count_distinct_desc.bitmap_sz_bits;
        } else {
          init_agg_vals_[agg_col_idx] = allocateCountDistinctBitmap(count_distinct_desc.bitmap_sz_bits);
        }
      } else {
        CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet);
        if (deferred) {
          agg_bitmap_size[agg_col_idx] = -1;
        } else {
          init_agg_vals_[agg_col_idx] = allocateCountDistinctSet();
        }
      }
    }
    if (agg_info.agg_kind == kAVG) {
      ++agg_col_idx;
    }
  }

  return agg_bitmap_size;
}

int64_t QueryExecutionContext::allocateCountDistinctBitmap(const size_t bitmap_sz) {
  auto bitmap_byte_sz = bitmap_size_bytes(bitmap_sz);
  auto count_distinct_buffer = static_cast<int8_t*>(checked_calloc(bitmap_byte_sz, 1));
  row_set_mem_owner_->addCountDistinctBuffer(count_distinct_buffer);
  return reinterpret_cast<int64_t>(count_distinct_buffer);
}

int64_t QueryExecutionContext::allocateCountDistinctSet() {
  auto count_distinct_set = new std::set<int64_t>();
  row_set_mem_owner_->addCountDistinctSet(count_distinct_set);
  return reinterpret_cast<int64_t>(count_distinct_set);
}

ResultRows QueryExecutionContext::getRowSet(const std::vector<Analyzer::Expr*>& targets,
                                            const QueryMemoryDescriptor& query_mem_desc,
                                            const bool was_auto_device) const noexcept {
  std::vector<std::pair<ResultRows, std::vector<size_t>>> results_per_sm;
  CHECK_EQ(num_buffers_, group_by_buffers_.size());
  if (device_type_ == ExecutorDeviceType::CPU) {
    CHECK_EQ(size_t(1), num_buffers_);
    return groupBufferToResults(0, targets, was_auto_device);
  }
  size_t step{query_mem_desc_.threadsShareMemory() ? executor_->blockSize() : 1};
  for (size_t i = 0; i < group_by_buffers_.size(); i += step) {
    results_per_sm.emplace_back(groupBufferToResults(i, targets, was_auto_device), std::vector<size_t>{});
  }
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return executor_->reduceMultiDeviceResults(results_per_sm, row_set_mem_owner_, query_mem_desc, output_columnar_);
}

bool QueryExecutionContext::isEmptyBin(const int64_t* group_by_buffer, const size_t bin, const size_t key_idx) const {
  const size_t key_off = query_mem_desc_.getKeyOffInBytes(bin, key_idx) / sizeof(int64_t);
  if (group_by_buffer[key_off] == EMPTY_KEY_64) {
    return true;
  }
  return false;
}

void QueryExecutionContext::outputBin(ResultRows& results,
                                      const std::vector<Analyzer::Expr*>& targets,
                                      int64_t* group_by_buffer,
                                      const size_t bin) const {
  if (isEmptyBin(group_by_buffer, bin, 0)) {
    return;
  }

  const size_t group_by_col_count{query_mem_desc_.group_col_widths.size()};
  size_t out_vec_idx = 0;
  int8_t* buffer_ptr = reinterpret_cast<int8_t*>(group_by_buffer) + query_mem_desc_.getKeyOffInBytes(bin);

  if (group_by_col_count > 1) {
    std::vector<int64_t> multi_key;
    CHECK(!output_columnar_);
    for (size_t key_idx = 0; key_idx < group_by_col_count; ++key_idx) {
      const auto key_comp = get_component(
          buffer_ptr,
          compact_byte_width(query_mem_desc_.group_col_widths[key_idx], unsigned(SMALLEST_BYTE_WIDTH_TO_COMPACT)));
      multi_key.push_back(key_comp);
      buffer_ptr += query_mem_desc_.getNextKeyOffInBytes(key_idx);
    }
    results.beginRow(multi_key);
  } else {
    const auto key_comp = get_component(buffer_ptr, sizeof(int64_t));
    results.beginRow(key_comp);
    buffer_ptr += query_mem_desc_.getNextKeyOffInBytes(0);
  }
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
        CHECK_EQ(size_t(1), col_buffers_.size());
        auto& frag_col_buffers = col_buffers_.front();
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
            switch (elem_ti.get_size()) {
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
          elem_sz = elem_ti.get_size();
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
        CHECK_EQ(size_t(1), col_buffers_.size());
        auto& frag_col_buffers = col_buffers_.front();
        val1 = lazy_decode(static_cast<Analyzer::ColumnVar*>(target_expr), frag_col_buffers[col_id], val1);
      }
      const auto agg_info = compact_target_info(target_expr);
      if (agg_info.sql_type.get_type() == kFLOAT && (is_lazy_fetched || chosen_byte_width == sizeof(float))) {
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

ResultRows QueryExecutionContext::groupBufferToResults(const size_t i,
                                                       const std::vector<Analyzer::Expr*>& targets,
                                                       const bool was_auto_device) const {
  const size_t group_by_col_count{query_mem_desc_.group_col_widths.size()};
  const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
  CHECK(!output_columnar_ || group_by_col_count == 1);
  auto impl = [group_by_col_count, agg_col_count, was_auto_device, this, &targets](
      const size_t groups_buffer_entry_count, int64_t* group_by_buffer) {
    if (query_mem_desc_.keyless_hash) {
      CHECK(!sort_on_gpu_);
      CHECK_EQ(size_t(1), group_by_col_count);
      const int8_t warp_count = query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1;
      if (!query_mem_desc_.interleavedBins(ExecutorDeviceType::GPU) || !was_auto_device) {
        return ResultRows(query_mem_desc_,
                          targets,
                          executor_,
                          row_set_mem_owner_,
                          device_type_,
                          group_by_buffer,
                          groups_buffer_entry_count,
                          query_mem_desc_.min_val,
                          warp_count);
      }
      // Can't do the fast reduction in auto mode for interleaved bins, warp count isn't the same
      ResultRows results({}, targets, executor_, row_set_mem_owner_, ExecutorDeviceType::CPU);
      results.addKeylessGroupByBuffer(
          group_by_buffer, groups_buffer_entry_count, query_mem_desc_.min_val, warp_count, output_columnar_);
      return results;
    }
    ResultRows results(query_mem_desc_,
                       targets,
                       row_set_mem_owner_,
                       group_by_buffer,
                       groups_buffer_entry_count,
                       output_columnar_,
                       col_buffers_,
                       device_type_,
                       device_id_);
    if (results.in_place_) {
      return results;
    }
    for (size_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
      outputBin(results, targets, group_by_buffer, bin);
    }
    return results;
  };
  ResultRows results(query_mem_desc_,
                     targets,
                     row_set_mem_owner_,
                     nullptr,
                     0,
                     output_columnar_,
                     col_buffers_,
                     device_type_,
                     device_id_);
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    CHECK(!sort_on_gpu_);
    results = impl(query_mem_desc_.entry_count_small, small_group_by_buffers_[i]);
  }
  CHECK_LT(i, group_by_buffers_.size());
  auto more_results = impl(query_mem_desc_.entry_count, group_by_buffers_[i]);
  if (query_mem_desc_.keyless_hash) {
    CHECK(!sort_on_gpu_);
    return more_results;
  }
  results.append(more_results);
  return results;
}

#ifdef HAVE_CUDA
std::vector<CUdeviceptr> QueryExecutionContext::prepareKernelParams(
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int8_t>& literal_buff,
    const std::vector<int64_t>& num_rows,
    const std::vector<uint64_t>& frag_row_offsets,
    const int32_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    const std::vector<int32_t>& error_codes,
    const unsigned grid_size_x,
    const uint32_t num_tables,
    const int64_t join_hash_table,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const bool hoist_literals,
    const bool is_group_by) const {
  std::vector<CUdeviceptr> params(KERN_PARAM_COUNT, 0);
  uint32_t num_fragments = col_buffers.size();
  const size_t col_count{num_fragments > 0 ? col_buffers.front().size() : 0};
  if (col_count) {
    std::vector<CUdeviceptr> multifrag_col_dev_buffers;
    for (auto frag_col_buffers : col_buffers) {
      std::vector<CUdeviceptr> col_dev_buffers;
      for (auto col_buffer : frag_col_buffers) {
        col_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(col_buffer));
      }
      auto col_buffers_dev_ptr = alloc_gpu_mem(data_mgr, col_count * sizeof(CUdeviceptr), device_id, nullptr);
      copy_to_gpu(data_mgr, col_buffers_dev_ptr, &col_dev_buffers[0], col_count * sizeof(CUdeviceptr), device_id);
      multifrag_col_dev_buffers.push_back(col_buffers_dev_ptr);
    }
    params[COL_BUFFERS] = alloc_gpu_mem(data_mgr, num_fragments * sizeof(CUdeviceptr), device_id, nullptr);
    copy_to_gpu(
        data_mgr, params[COL_BUFFERS], &multifrag_col_dev_buffers[0], num_fragments * sizeof(CUdeviceptr), device_id);
  }
  params[NUM_FRAGMENTS] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[NUM_FRAGMENTS], &num_fragments, sizeof(uint32_t), device_id);
  if (!literal_buff.empty()) {
    CHECK(hoist_literals);
    params[LITERALS] = alloc_gpu_mem(data_mgr, literal_buff.size(), device_id, nullptr);
    copy_to_gpu(data_mgr, params[LITERALS], &literal_buff[0], literal_buff.size(), device_id);
  }
  params[NUM_ROWS] = alloc_gpu_mem(data_mgr, sizeof(int64_t) * num_rows.size(), device_id, nullptr);
  copy_to_gpu(data_mgr, params[NUM_ROWS], &num_rows[0], sizeof(int64_t) * num_rows.size(), device_id);
  params[FRAG_ROW_OFFSETS] = alloc_gpu_mem(data_mgr, sizeof(int64_t) * frag_row_offsets.size(), device_id, nullptr);
  copy_to_gpu(
      data_mgr, params[FRAG_ROW_OFFSETS], &frag_row_offsets[0], sizeof(int64_t) * frag_row_offsets.size(), device_id);
  int32_t max_matched{scan_limit};
  params[MAX_MATCHED] = alloc_gpu_mem(data_mgr, sizeof(max_matched), device_id, nullptr);
  copy_to_gpu(data_mgr, params[MAX_MATCHED], &max_matched, sizeof(max_matched), device_id);

  int32_t total_matched{0};
  params[TOTAL_MATCHED] = alloc_gpu_mem(data_mgr, sizeof(total_matched), device_id, nullptr);
  copy_to_gpu(data_mgr, params[TOTAL_MATCHED], &total_matched, sizeof(total_matched), device_id);

  if (is_group_by && !output_columnar_) {
    auto cmpt_sz = align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t);
    auto cmpt_val_buff = compact_init_vals(cmpt_sz, init_agg_vals, query_mem_desc_.agg_col_widths);
    params[INIT_AGG_VALS] = alloc_gpu_mem(data_mgr, cmpt_sz * sizeof(int64_t), device_id, nullptr);
    copy_to_gpu(data_mgr, params[INIT_AGG_VALS], &cmpt_val_buff[0], cmpt_sz * sizeof(int64_t), device_id);
  } else {
    params[INIT_AGG_VALS] = alloc_gpu_mem(data_mgr, init_agg_vals.size() * sizeof(int64_t), device_id, nullptr);
    copy_to_gpu(data_mgr, params[INIT_AGG_VALS], &init_agg_vals[0], init_agg_vals.size() * sizeof(int64_t), device_id);
  }

  params[ERROR_CODE] = alloc_gpu_mem(data_mgr, grid_size_x * sizeof(error_codes[0]), device_id, nullptr);
  copy_to_gpu(data_mgr, params[ERROR_CODE], &error_codes[0], grid_size_x * sizeof(error_codes[0]), device_id);

  params[NUM_TABLES] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[NUM_TABLES], &num_tables, sizeof(uint32_t), device_id);

  params[JOIN_HASH_TABLE] = alloc_gpu_mem(data_mgr, sizeof(int64_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[JOIN_HASH_TABLE], &join_hash_table, sizeof(int64_t), device_id);

  return params;
}

GpuQueryMemory QueryExecutionContext::prepareGroupByDevBuffer(Data_Namespace::DataMgr* data_mgr,
                                                              RenderAllocator* render_allocator,
                                                              const CUdeviceptr init_agg_vals_dev_ptr,
                                                              const int device_id,
                                                              const unsigned block_size_x,
                                                              const unsigned grid_size_x,
                                                              const bool can_sort_on_gpu) const {
  auto gpu_query_mem = create_dev_group_by_buffers(data_mgr,
                                                   group_by_buffers_,
                                                   small_group_by_buffers_,
                                                   query_mem_desc_,
                                                   block_size_x,
                                                   grid_size_x,
                                                   device_id,
                                                   can_sort_on_gpu,
                                                   false,
                                                   render_allocator);
  if (render_allocator) {
    CHECK_EQ(size_t(0), render_allocator->getAllocatedSize() % 8);
  }
  if (query_mem_desc_.lazyInitGroups(ExecutorDeviceType::GPU) &&
      query_mem_desc_.hash_type != GroupByColRangeType::MultiCol) {
    CHECK(!render_allocator);
    const size_t step{query_mem_desc_.threadsShareMemory() ? block_size_x : 1};
    size_t groups_buffer_size{query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU)};
    auto group_by_dev_buffer = gpu_query_mem.group_by_buffers.second;
    const size_t col_count = query_mem_desc_.agg_col_widths.size();
    CUdeviceptr col_widths_dev_ptr{0};
    if (output_columnar_) {
      std::vector<int8_t> compact_col_widths(col_count);
      for (size_t idx = 0; idx < col_count; ++idx) {
        compact_col_widths[idx] = query_mem_desc_.agg_col_widths[idx].compact;
      }
      col_widths_dev_ptr = alloc_gpu_mem(data_mgr, col_count * sizeof(int8_t), device_id, nullptr);
      copy_to_gpu(data_mgr, col_widths_dev_ptr, &compact_col_widths[0], col_count * sizeof(int8_t), device_id);
    }
    const int8_t warp_count = query_mem_desc_.interleavedBins(ExecutorDeviceType::GPU) ? executor_->warpSize() : 1;
    for (size_t i = 0; i < group_by_buffers_.size(); i += step) {
      if (output_columnar_) {
        init_columnar_group_by_buffer_on_device(reinterpret_cast<int64_t*>(group_by_dev_buffer),
                                                reinterpret_cast<const int64_t*>(init_agg_vals_dev_ptr),
                                                query_mem_desc_.entry_count,
                                                query_mem_desc_.group_col_widths.size(),
                                                col_count,
                                                reinterpret_cast<int8_t*>(col_widths_dev_ptr),
                                                query_mem_desc_.keyless_hash,
                                                sizeof(int64_t),
                                                block_size_x,
                                                grid_size_x);
      } else {
        init_group_by_buffer_on_device(reinterpret_cast<int64_t*>(group_by_dev_buffer),
                                       reinterpret_cast<int64_t*>(init_agg_vals_dev_ptr),
                                       query_mem_desc_.entry_count,
                                       query_mem_desc_.group_col_widths.size(),
                                       query_mem_desc_.getRowSize() / sizeof(int64_t),
                                       query_mem_desc_.keyless_hash,
                                       warp_count,
                                       block_size_x,
                                       grid_size_x);
      }
      group_by_dev_buffer += groups_buffer_size;
    }
  }
  return gpu_query_mem;
}
#endif

std::vector<int64_t*> QueryExecutionContext::launchGpuCode(const std::vector<void*>& cu_functions,
                                                           const bool hoist_literals,
                                                           const std::vector<int8_t>& literal_buff,
                                                           std::vector<std::vector<const int8_t*>> col_buffers,
                                                           const std::vector<int64_t>& num_rows,
                                                           const std::vector<uint64_t>& frag_row_offsets,
                                                           const int32_t scan_limit,
                                                           const std::vector<int64_t>& init_agg_vals,
                                                           Data_Namespace::DataMgr* data_mgr,
                                                           const unsigned block_size_x,
                                                           const unsigned grid_size_x,
                                                           const int device_id,
                                                           int32_t* error_code,
                                                           const uint32_t num_tables,
                                                           const int64_t join_hash_table,
                                                           RenderAllocatorMap* render_allocator_map) const {
#ifdef HAVE_CUDA
  bool is_group_by{query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU) > 0};
  data_mgr->cudaMgr_->setContext(device_id);

  RenderAllocator* render_allocator = nullptr;
  if (render_allocator_map) {
    render_allocator = render_allocator_map->getRenderAllocator(device_id);
  }

  auto cu_func = static_cast<CUfunction>(cu_functions[device_id]);
  std::vector<int64_t*> out_vec;
  uint32_t num_fragments = col_buffers.size();
  std::vector<int32_t> error_codes(block_size_x);

  auto kernel_params = prepareKernelParams(col_buffers,
                                           literal_buff,
                                           num_rows,
                                           frag_row_offsets,
                                           scan_limit,
                                           init_agg_vals,
                                           error_codes,
                                           grid_size_x,
                                           num_tables,
                                           join_hash_table,
                                           data_mgr,
                                           device_id,
                                           hoist_literals,
                                           is_group_by);

  CHECK_EQ(static_cast<size_t>(KERN_PARAM_COUNT), kernel_params.size());
  CHECK_EQ(CUdeviceptr(0), kernel_params[GROUPBY_BUF]);
  CHECK_EQ(CUdeviceptr(0), kernel_params[SMALL_BUF]);

  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;
  if (is_group_by) {
    CHECK(!group_by_buffers_.empty() || render_allocator);
    bool can_sort_on_gpu = query_mem_desc_.sortOnGpu();
    auto gpu_query_mem = prepareGroupByDevBuffer(data_mgr,
                                                 render_allocator,
                                                 kernel_params[INIT_AGG_VALS],
                                                 device_id,
                                                 block_size_x,
                                                 grid_size_x,
                                                 can_sort_on_gpu);

    kernel_params[GROUPBY_BUF] = gpu_query_mem.group_by_buffers.first;
    kernel_params[SMALL_BUF] = gpu_query_mem.small_group_by_buffers.first;
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }
    if (hoist_literals) {
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    } else {
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    }
    if (!render_allocator) {
      copy_group_by_buffers_from_gpu(data_mgr,
                                     this,
                                     gpu_query_mem,
                                     block_size_x,
                                     grid_size_x,
                                     device_id,
                                     can_sort_on_gpu && query_mem_desc_.keyless_hash);
    }
    copy_from_gpu(
        data_mgr, &error_codes[0], kernel_params[ERROR_CODE], grid_size_x * sizeof(error_codes[0]), device_id);
    *error_code = 0;
    for (const auto err : error_codes) {
      if (err && (!*error_code || err > *error_code)) {
        *error_code = err;
        break;
      }
    }
  } else {
    std::vector<CUdeviceptr> out_vec_dev_buffers;
    const size_t agg_col_count{init_agg_vals.size()};
    for (size_t i = 0; i < agg_col_count; ++i) {
      auto out_vec_dev_buffer =
          num_fragments
              ? alloc_gpu_mem(
                    data_mgr, block_size_x * grid_size_x * sizeof(int64_t) * num_fragments, device_id, nullptr)
              : 0;
      out_vec_dev_buffers.push_back(out_vec_dev_buffer);
    }
    auto out_vec_dev_ptr = alloc_gpu_mem(data_mgr, agg_col_count * sizeof(CUdeviceptr), device_id, nullptr);
    copy_to_gpu(data_mgr, out_vec_dev_ptr, &out_vec_dev_buffers[0], agg_col_count * sizeof(CUdeviceptr), device_id);
    CUdeviceptr unused_dev_ptr{0};
    kernel_params[GROUPBY_BUF] = out_vec_dev_ptr;
    kernel_params[SMALL_BUF] = unused_dev_ptr;
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }
    if (hoist_literals) {
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     0,
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    } else {
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     0,
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    }
    for (size_t i = 0; i < agg_col_count; ++i) {
      int64_t* host_out_vec = new int64_t[block_size_x * grid_size_x * sizeof(int64_t) * num_fragments];
      copy_from_gpu(data_mgr,
                    host_out_vec,
                    out_vec_dev_buffers[i],
                    block_size_x * grid_size_x * sizeof(int64_t) * num_fragments,
                    device_id);
      out_vec.push_back(host_out_vec);
    }
  }
  return out_vec;
#else
  return {};
#endif
}

std::unique_ptr<QueryExecutionContext> QueryMemoryDescriptor::getQueryExecutionContext(
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const int device_id,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool output_columnar,
    const bool sort_on_gpu,
    RenderAllocatorMap* render_allocator_map) const {
  return std::unique_ptr<QueryExecutionContext>(new QueryExecutionContext(*this,
                                                                          init_agg_vals,
                                                                          executor,
                                                                          device_type,
                                                                          device_id,
                                                                          col_buffers,
                                                                          row_set_mem_owner,
                                                                          output_columnar,
                                                                          sort_on_gpu,
                                                                          render_allocator_map));
}

size_t QueryMemoryDescriptor::getColsSize() const {
  CHECK(!output_columnar);
  size_t total_bytes{0};
  for (size_t col_idx = 0; col_idx < agg_col_widths.size(); ++col_idx) {
    auto chosen_bytes = agg_col_widths[col_idx].compact;
    if (chosen_bytes == sizeof(int64_t)) {
      total_bytes = align_to_int64(total_bytes);
    }
    total_bytes += chosen_bytes;
  }
  return total_bytes;
}

size_t QueryMemoryDescriptor::getRowSize() const {
  CHECK(!output_columnar);
  size_t total_bytes{0};
  if (keyless_hash) {
    CHECK_EQ(size_t(1), group_col_widths.size());
  } else {
    total_bytes += group_col_widths.size() * sizeof(int64_t);
  }
  total_bytes += getColsSize();
  return align_to_int64(total_bytes);
}

size_t QueryMemoryDescriptor::getWarpCount() const {
  return (interleaved_bins_on_gpu ? executor_->warpSize() : 1);
}

size_t QueryMemoryDescriptor::getTotalBytesOfColumnarBuffers(const std::vector<ColWidths>& col_widths) const {
  CHECK(output_columnar);
  size_t total_bytes{0};
  for (size_t col_idx = 0; col_idx < col_widths.size(); ++col_idx) {
    total_bytes += col_widths[col_idx].compact * entry_count;
    total_bytes = align_to_int64(total_bytes);
  }
  return total_bytes;
}

size_t QueryMemoryDescriptor::getKeyOffInBytes(const size_t bin, const size_t key_idx) const {
  CHECK(!keyless_hash);
  if (output_columnar) {
    CHECK_EQ(size_t(0), key_idx);
    return bin * sizeof(int64_t);
  }

  CHECK_LT(key_idx, group_col_widths.size());
  auto offset = bin * getRowSize();
  CHECK_EQ(size_t(0), offset % sizeof(int64_t));
  offset += key_idx * sizeof(int64_t);
  return offset;
}

size_t QueryMemoryDescriptor::getNextKeyOffInBytes(const size_t crt_idx) const {
  CHECK(!keyless_hash);
  CHECK_LT(crt_idx, group_col_widths.size());
  if (output_columnar) {
    CHECK_EQ(size_t(0), crt_idx);
  }
  return sizeof(int64_t);
}

size_t QueryMemoryDescriptor::getColOnlyOffInBytes(const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  size_t offset{0};
  for (size_t index = 0; index < col_idx; ++index) {
    const auto chosen_bytes = agg_col_widths[index].compact;
    if (chosen_bytes == sizeof(int64_t)) {
      offset = align_to_int64(offset);
    }
    offset += chosen_bytes;
  }

  if (sizeof(int64_t) == agg_col_widths[col_idx].compact) {
    offset = align_to_int64(offset);
  }

  return offset;
}

size_t QueryMemoryDescriptor::getColOffInBytes(const size_t bin, const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  auto warp_count = getWarpCount();
  if (output_columnar) {
    CHECK_LT(bin, entry_count);
    CHECK_EQ(size_t(1), group_col_widths.size());
    CHECK_EQ(size_t(1), warp_count);
    size_t offset{0};
    if (!keyless_hash) {
      offset = sizeof(int64_t) * entry_count;
    }
    for (size_t index = 0; index < col_idx; ++index) {
      offset += agg_col_widths[index].compact * entry_count;
      offset = align_to_int64(offset);
    }
    offset += bin * agg_col_widths[col_idx].compact;
    return offset;
  }

  auto offset = bin * warp_count * getRowSize();
  if (keyless_hash) {
    CHECK_EQ(size_t(1), group_col_widths.size());
  } else {
    offset += group_col_widths.size() * sizeof(int64_t);
  }
  offset += getColOnlyOffInBytes(col_idx);
  return offset;
}

size_t QueryMemoryDescriptor::getColOffInBytesInNextBin(const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  auto warp_count = getWarpCount();
  if (output_columnar) {
    CHECK_EQ(size_t(1), group_col_widths.size());
    CHECK_EQ(size_t(1), warp_count);
    return agg_col_widths[col_idx].compact;
  }

  return warp_count * getRowSize();
}

size_t QueryMemoryDescriptor::getNextColOffInBytes(const int8_t* col_ptr,
                                                   const size_t bin,
                                                   const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  CHECK(!output_columnar || bin < entry_count);
  size_t offset{0};
  auto warp_count = getWarpCount();
  const auto chosen_bytes = agg_col_widths[col_idx].compact;
  if (col_idx + 1 == agg_col_widths.size()) {
    if (output_columnar) {
      return (entry_count - bin) * chosen_bytes;
    } else {
      return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
    }
  }

  const auto next_chosen_bytes = agg_col_widths[col_idx + 1].compact;
  if (output_columnar) {
    CHECK_EQ(size_t(1), group_col_widths.size());
    CHECK_EQ(size_t(1), warp_count);

    offset = entry_count * chosen_bytes;
    offset = align_to_int64(offset);
    offset += bin * (next_chosen_bytes - chosen_bytes);
    return offset;
  }

  if (next_chosen_bytes == sizeof(int64_t)) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  } else {
    return chosen_bytes;
  }
}

size_t QueryMemoryDescriptor::getBufferSizeQuad(const ExecutorDeviceType device_type) const {
  if (keyless_hash) {
    CHECK_EQ(size_t(1), group_col_widths.size());
    auto total_bytes = align_to_int64(getColsSize());

    return (interleavedBins(device_type) ? executor_->warpSize() : 1) * entry_count * total_bytes / sizeof(int64_t);
  }

  size_t total_bytes{0};
  if (output_columnar) {
    CHECK_EQ(size_t(1), group_col_widths.size());
    total_bytes = sizeof(int64_t) * entry_count + getTotalBytesOfColumnarBuffers(agg_col_widths);
  } else {
    total_bytes = getRowSize() * entry_count;
  }

  return total_bytes / sizeof(int64_t);
}

size_t QueryMemoryDescriptor::getSmallBufferSizeQuad() const {
  CHECK(!keyless_hash || entry_count_small == 0);
  return (group_col_widths.size() + agg_col_widths.size()) * entry_count_small;
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(const ExecutorDeviceType device_type) const {
  return getBufferSizeQuad(device_type) * sizeof(int64_t);
}

size_t QueryMemoryDescriptor::getSmallBufferSizeBytes() const {
  return getSmallBufferSizeQuad() * sizeof(int64_t);
}

namespace {

int32_t get_agg_count(const std::vector<Analyzer::Expr*>& target_exprs) {
  int32_t agg_count{0};
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      const auto& ti = target_expr->get_type_info();
      if (ti.is_string() && ti.get_compression() != kENCODING_DICT) {
        agg_count += 2;
      } else {
        ++agg_count;
      }
      continue;
    }
    if (agg_expr && agg_expr->get_aggtype() == kAVG) {
      agg_count += 2;
    } else {
      ++agg_count;
    }
  }
  return agg_count;
}

}  // namespace

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getColRangeInfo() {
  if (ra_exe_unit_.groupby_exprs.size() != 1) {
    try {
      checked_int64_t cardinality{1};
      bool has_nulls{false};
      for (const auto groupby_expr : ra_exe_unit_.groupby_exprs) {
        auto col_range_info = getExprRangeInfo(groupby_expr.get());
        if (col_range_info.hash_type_ != GroupByColRangeType::OneColKnownRange) {
          return {GroupByColRangeType::MultiCol, 0, 0, 0, false};
        }
        auto crt_col_cardinality = col_range_info.max - col_range_info.min + 1 + (col_range_info.has_nulls ? 1 : 0);
        CHECK_GT(crt_col_cardinality, 0);
        cardinality *= crt_col_cardinality;
        if (col_range_info.has_nulls) {
          has_nulls = true;
        }
      }
      if (cardinality > 10000000) {  // more than 10M groups is a lot
        return {GroupByColRangeType::MultiCol, 0, 0, 0, false};
      }
      return {GroupByColRangeType::MultiColPerfectHash, 0, int64_t(cardinality), 0, has_nulls};
    } catch (...) {  // overflow when computing cardinality
      return {GroupByColRangeType::MultiCol, 0, 0, 0, false};
    }
  }
  return getExprRangeInfo(ra_exe_unit_.groupby_exprs.front().get());
}

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getExprRangeInfo(const Analyzer::Expr* expr) const {
  const int64_t guessed_range_max{255};  // TODO(alex): replace with educated guess

  const auto expr_range = getExpressionRange(expr, query_infos_, executor_);
  switch (expr_range.getType()) {
    case ExpressionRangeType::Integer:
      return {GroupByColRangeType::OneColKnownRange,
              expr_range.getIntMin(),
              expr_range.getIntMax(),
              expr_range.getBucket(),
              expr_range.hasNulls()};
    case ExpressionRangeType::FloatingPoint:
      if (g_enable_watchdog) {
        throw WatchdogException("Group by float / double would be slow");
      }
    case ExpressionRangeType::Invalid:
      return {GroupByColRangeType::OneColGuessedRange, 0, guessed_range_max, 0, false};
    default:
      CHECK(false);
  }
  CHECK(false);
  return {GroupByColRangeType::Scan, 0, 0, 0, false};
}

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->ll_int(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

namespace {

bool many_entries(const int64_t max_val, const int64_t min_val, const int64_t bucket) {
  return max_val - min_val > 10000 * std::max(bucket, int64_t(1));
}

}  // namespace

GroupByAndAggregate::GroupByAndAggregate(Executor* executor,
                                         const ExecutorDeviceType device_type,
                                         const RelAlgExecutionUnit& ra_exe_unit,
                                         const bool render_output,
                                         const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                         std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                         const size_t max_groups_buffer_entry_count,
                                         const size_t small_groups_buffer_entry_count,
                                         const bool allow_multifrag,
                                         const bool output_columnar_hint)
    : executor_(executor), ra_exe_unit_(ra_exe_unit), query_infos_(query_infos), row_set_mem_owner_(row_set_mem_owner) {
  for (const auto groupby_expr : ra_exe_unit.groupby_exprs) {
    if (!groupby_expr) {
      continue;
    }
    const auto& groupby_ti = groupby_expr->get_type_info();
    if (groupby_ti.is_string() && groupby_ti.get_compression() != kENCODING_DICT) {
      throw std::runtime_error("Cannot group by string columns which are not dictionary encoded.");
    }
  }
  bool sort_on_gpu_hint = device_type == ExecutorDeviceType::GPU && allow_multifrag &&
                          !ra_exe_unit.order_entries.empty() && gpuCanHandleOrderEntries(ra_exe_unit.order_entries);
  initQueryMemoryDescriptor(
      allow_multifrag, max_groups_buffer_entry_count, small_groups_buffer_entry_count, sort_on_gpu_hint, render_output);
  if (device_type != ExecutorDeviceType::GPU) {
    // TODO(miyu): remove w/ interleaving
    query_mem_desc_.interleaved_bins_on_gpu = false;
  }
  query_mem_desc_.sort_on_gpu_ =
      sort_on_gpu_hint && query_mem_desc_.canOutputColumnar() && !query_mem_desc_.keyless_hash;
  query_mem_desc_.is_sort_plan = !ra_exe_unit.order_entries.empty() && !query_mem_desc_.sort_on_gpu_;
  output_columnar_ = (output_columnar_hint && query_mem_desc_.canOutputColumnar()) || query_mem_desc_.sortOnGpu();
  query_mem_desc_.output_columnar = output_columnar_;
}

void GroupByAndAggregate::initQueryMemoryDescriptor(const bool allow_multifrag,
                                                    const size_t max_groups_buffer_entry_count,
                                                    const size_t small_groups_buffer_entry_count,
                                                    const bool sort_on_gpu_hint,
                                                    const bool render_output) {
  addTransientStringLiterals();

  const auto count_distinct_descriptors = initCountDistinctDescriptors();
  if (!count_distinct_descriptors.empty()) {
    CHECK(row_set_mem_owner_);
    row_set_mem_owner_->setCountDistinctDescriptors(count_distinct_descriptors);
  }

  std::vector<ColWidths> agg_col_widths;
  for (auto wid : get_col_byte_widths(ra_exe_unit_.target_exprs)) {
    agg_col_widths.push_back({wid, int8_t(compact_byte_width(wid, unsigned(SMALLEST_BYTE_WIDTH_TO_COMPACT)))});
  }
  auto group_col_widths = get_col_byte_widths(ra_exe_unit_.groupby_exprs);

  const bool is_group_by{!group_col_widths.empty()};
  if (!is_group_by) {
    CHECK(!render_output);
    query_mem_desc_ = {executor_,
                       allow_multifrag,
                       GroupByColRangeType::Scan,
                       false,
                       false,
                       -1,
                       0,
                       group_col_widths,
                       agg_col_widths,
                       0,
                       0,
                       0,
                       0,
                       0,
                       false,
                       GroupByMemSharing::Private,
                       count_distinct_descriptors,
                       false,
                       false,
                       false,
                       false};
    return;
  }

  const auto col_range_info = getColRangeInfo();

  if (g_enable_watchdog && col_range_info.hash_type_ != GroupByColRangeType::OneColKnownRange &&
      col_range_info.hash_type_ != GroupByColRangeType::MultiColPerfectHash &&
      col_range_info.hash_type_ != GroupByColRangeType::OneColGuessedRange && !render_output &&
      (ra_exe_unit_.scan_limit == 0 || ra_exe_unit_.scan_limit > 10000)) {
    throw WatchdogException("Query would use too much memory");
  }

  switch (col_range_info.hash_type_) {
    case GroupByColRangeType::OneColKnownRange:
    case GroupByColRangeType::OneColGuessedRange:
    case GroupByColRangeType::Scan: {
      if (col_range_info.hash_type_ == GroupByColRangeType::OneColGuessedRange ||
          col_range_info.hash_type_ == GroupByColRangeType::Scan ||
          ((ra_exe_unit_.groupby_exprs.size() != 1 ||
            !ra_exe_unit_.groupby_exprs.front()->get_type_info().is_string()) &&
           col_range_info.max >= col_range_info.min + static_cast<int64_t>(max_groups_buffer_entry_count) &&
           !col_range_info.bucket)) {
        const auto hash_type = render_output ? GroupByColRangeType::MultiCol : col_range_info.hash_type_;
        size_t small_group_slots =
            ra_exe_unit_.scan_limit ? static_cast<size_t>(ra_exe_unit_.scan_limit) : small_groups_buffer_entry_count;
        if (render_output) {
          small_group_slots = 0;
        }
        query_mem_desc_ = {executor_,
                           allow_multifrag,
                           hash_type,
                           false,
                           false,
                           -1,
                           0,
                           group_col_widths,
                           agg_col_widths,
                           max_groups_buffer_entry_count * (render_output ? 4 : 1),
                           small_group_slots,
                           col_range_info.min,
                           col_range_info.max,
                           0,
                           col_range_info.has_nulls,
                           GroupByMemSharing::Shared,
                           count_distinct_descriptors,
                           false,
                           false,
                           false,
                           render_output};
        return;
      } else {
        CHECK(!render_output);
        const auto keyless_info = getKeylessInfo(ra_exe_unit_.target_exprs, is_group_by);
        bool keyless =
            (!sort_on_gpu_hint || !many_entries(col_range_info.max, col_range_info.min, col_range_info.bucket)) &&
            !col_range_info.bucket && keyless_info.keyless;
        size_t bin_count = col_range_info.max - col_range_info.min;
        if (col_range_info.bucket) {
          bin_count /= col_range_info.bucket;
        }
        bin_count += (1 + (col_range_info.has_nulls ? 1 : 0));
        const size_t interleaved_max_threshold{20};
        bool interleaved_bins = keyless && (bin_count <= interleaved_max_threshold);
        query_mem_desc_ = {executor_,
                           allow_multifrag,
                           col_range_info.hash_type_,
                           keyless,
                           interleaved_bins,
                           keyless_info.target_index,
                           keyless_info.init_val,
                           group_col_widths,
                           agg_col_widths,
                           bin_count,
                           0,
                           col_range_info.min,
                           col_range_info.max,
                           col_range_info.bucket,
                           col_range_info.has_nulls,
                           GroupByMemSharing::Shared,
                           count_distinct_descriptors,
                           false,
                           false,
                           false,
                           false};
        return;
      }
    }
    case GroupByColRangeType::MultiCol: {
      CHECK(!render_output);
      query_mem_desc_ = {executor_,
                         allow_multifrag,
                         col_range_info.hash_type_,
                         false,
                         false,
                         -1,
                         0,
                         group_col_widths,
                         agg_col_widths,
                         max_groups_buffer_entry_count,
                         0,
                         0,
                         0,
                         0,
                         false,
                         GroupByMemSharing::Shared,
                         count_distinct_descriptors,
                         false,
                         false,
                         false,
                         false};
      return;
    }
    case GroupByColRangeType::MultiColPerfectHash: {
      CHECK(!render_output);
      query_mem_desc_ = {executor_,
                         allow_multifrag,
                         col_range_info.hash_type_,
                         false,
                         false,
                         -1,
                         0,
                         group_col_widths,
                         agg_col_widths,
                         static_cast<size_t>(col_range_info.max),
                         0,
                         col_range_info.min,
                         col_range_info.max,
                         0,
                         col_range_info.has_nulls,
                         GroupByMemSharing::Shared,
                         count_distinct_descriptors,
                         false,
                         false,
                         false,
                         false};
      return;
    }
    default:
      CHECK(false);
  }
  CHECK(false);
  return;
}

void GroupByAndAggregate::addTransientStringLiterals() {
  for (const auto group_expr : ra_exe_unit_.groupby_exprs) {
    if (!group_expr) {
      continue;
    }
    const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(group_expr.get());
    const auto& group_ti = group_expr->get_type_info();
    if (cast_expr && cast_expr->get_optype() == kCAST && group_ti.is_string()) {
      CHECK_EQ(kENCODING_DICT, group_ti.get_compression());
      auto sd = executor_->getStringDictionary(group_ti.get_comp_param(), row_set_mem_owner_);
      CHECK(sd);
      const auto str_lit_expr = dynamic_cast<const Analyzer::Constant*>(cast_expr->get_operand());
      if (str_lit_expr && str_lit_expr->get_constval().stringval) {
        sd->getOrAddTransient(*str_lit_expr->get_constval().stringval);
      }
      continue;
    }
    const auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(group_expr.get());
    if (!case_expr) {
      continue;
    }
    Analyzer::DomainSet domain_set;
    case_expr->get_domain(domain_set);
    if (domain_set.empty()) {
      continue;
    }
    if (group_ti.is_string()) {
      CHECK_EQ(kENCODING_DICT, group_ti.get_compression());
      auto sd = executor_->getStringDictionary(group_ti.get_comp_param(), row_set_mem_owner_);
      CHECK(sd);
      for (const auto domain_expr : domain_set) {
        const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(domain_expr);
        const auto str_lit_expr = cast_expr && cast_expr->get_optype() == kCAST
                                      ? dynamic_cast<const Analyzer::Constant*>(cast_expr->get_operand())
                                      : dynamic_cast<const Analyzer::Constant*>(domain_expr);
        if (str_lit_expr && str_lit_expr->get_constval().stringval) {
          sd->getOrAddTransient(*str_lit_expr->get_constval().stringval);
        }
      }
    }
  }
}

CountDistinctDescriptors GroupByAndAggregate::initCountDistinctDescriptors() {
  CountDistinctDescriptors count_distinct_descriptors;
  size_t target_idx{0};
  for (const auto target_expr : ra_exe_unit_.target_exprs) {
    auto agg_info = target_info(target_expr);
    if (agg_info.is_distinct) {
      CHECK(agg_info.is_agg);
      CHECK_EQ(kCOUNT, agg_info.agg_kind);
      const auto agg_expr = static_cast<const Analyzer::AggExpr*>(target_expr);
      const auto& arg_ti = agg_expr->get_arg()->get_type_info();
      if (arg_ti.is_string() && arg_ti.get_compression() != kENCODING_DICT) {
        throw std::runtime_error("Strings must be dictionary-encoded in COUNT(DISTINCT).");
      }
      auto arg_range_info = getExprRangeInfo(agg_expr->get_arg());
      CountDistinctImplType count_distinct_impl_type{CountDistinctImplType::StdSet};
      int64_t bitmap_sz_bits{0};
      if (arg_range_info.hash_type_ == GroupByColRangeType::OneColKnownRange &&
          !arg_ti.is_array()) {  // TODO(alex): allow bitmap implementation for arrays
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
        bitmap_sz_bits = arg_range_info.max - arg_range_info.min + 1;
        const int64_t MAX_BITMAP_BITS{8 * 1000 * 1000 * 1000L};
        if (bitmap_sz_bits <= 0 || bitmap_sz_bits > MAX_BITMAP_BITS) {
          count_distinct_impl_type = CountDistinctImplType::StdSet;
        }
      }
      if (g_enable_watchdog && count_distinct_impl_type == CountDistinctImplType::StdSet) {
        throw WatchdogException("Cannot use a fast path for COUNT distinct");
      }
      CountDistinctDescriptor count_distinct_desc{
          executor_, count_distinct_impl_type, arg_range_info.min, bitmap_sz_bits};
      auto it_ok = count_distinct_descriptors.insert(std::make_pair(target_idx, count_distinct_desc));
      CHECK(it_ok.second);
    }
    ++target_idx;
  }
  return count_distinct_descriptors;
}

const QueryMemoryDescriptor& GroupByAndAggregate::getQueryMemoryDescriptor() const {
  return query_mem_desc_;
}

bool GroupByAndAggregate::outputColumnar() const {
  return output_columnar_;
}

GroupByAndAggregate::KeylessInfo GroupByAndAggregate::getKeylessInfo(
    const std::vector<Analyzer::Expr*>& target_expr_list,
    const bool is_group_by) const {
  bool keyless{true}, found{false};
  int32_t index{0};
  int64_t init_val{0};
  for (const auto target_expr : target_expr_list) {
    auto agg_info = compact_target_info(target_expr);
    if (!found && agg_info.is_agg) {
      auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
      CHECK(agg_expr);
      const auto arg_expr = agg_arg(target_expr);
      switch (agg_info.agg_kind) {
        case kAVG:
          ++index;
          init_val = 0;
          found = true;
          break;
        case kCOUNT:
          if (arg_expr && !arg_expr->get_type_info().get_notnull()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos_, executor_);
            if (expr_range_info.hasNulls()) {
              break;
            }
          }
          init_val = 0;
          found = true;
          break;
        case kSUM: {
          if (!arg_expr->get_type_info().get_notnull()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos_, executor_);
            if (!expr_range_info.hasNulls()) {
              init_val = get_agg_initial_val(agg_info.agg_kind, arg_expr->get_type_info(), is_group_by);
              found = true;
            }
          } else {
            init_val = 0;
            auto expr_range_info = getExpressionRange(arg_expr, query_infos_, executor_);
            switch (expr_range_info.getType()) {
              case ExpressionRangeType::FloatingPoint:
                if (expr_range_info.getFpMax() < 0 || expr_range_info.getFpMin() > 0) {
                  found = true;
                }
                break;
              case ExpressionRangeType::Integer:
                if (expr_range_info.getIntMax() < 0 || expr_range_info.getIntMin() > 0) {
                  found = true;
                }
                break;
              default:
                break;
            }
          }
          break;
        }
        case kMIN: {
          auto expr_range_info = getExpressionRange(agg_expr->get_arg(), query_infos_, executor_);
          auto init_max = get_agg_initial_val(agg_info.agg_kind, agg_info.sql_type, is_group_by);
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::FloatingPoint: {
              init_val = init_max;
              auto double_max = *reinterpret_cast<const double*>(&init_max);
              if (expr_range_info.getFpMax() < double_max) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              init_val = init_max;
              if (expr_range_info.getIntMax() < init_max) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        case kMAX: {
          auto expr_range_info = getExpressionRange(agg_expr->get_arg(), query_infos_, executor_);
          auto init_min = get_agg_initial_val(agg_info.agg_kind, agg_info.sql_type, is_group_by);
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::FloatingPoint: {
              init_val = init_min;
              auto double_min = *reinterpret_cast<const double*>(&init_min);
              if (expr_range_info.getFpMin() > double_min) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              init_val = init_min;
              if (expr_range_info.getIntMin() > init_min) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        default:
          keyless = false;
          break;
      }
    }
    if (!keyless) {
      break;
    }
    if (!found) {
      ++index;
    }
  }

  // shouldn't use keyless for projection only
  return {keyless && found, index, init_val};
}

bool GroupByAndAggregate::gpuCanHandleOrderEntries(const std::list<Analyzer::OrderEntry>& order_entries) {
  if (order_entries.size() > 1) {  // TODO(alex): lift this restriction
    return false;
  }
  for (const auto order_entry : order_entries) {
    CHECK_GE(order_entry.tle_no, 1);
    CHECK_LE(static_cast<size_t>(order_entry.tle_no), ra_exe_unit_.target_exprs.size());
    const auto target_expr = ra_exe_unit_.target_exprs[order_entry.tle_no - 1];
    if (!dynamic_cast<Analyzer::AggExpr*>(target_expr)) {
      return false;
    }
    // TODO(alex): relax the restrictions
    auto agg_expr = static_cast<Analyzer::AggExpr*>(target_expr);
    if (agg_expr->get_is_distinct() || agg_expr->get_aggtype() == kAVG || agg_expr->get_aggtype() == kMIN ||
        agg_expr->get_aggtype() == kMAX) {
      return false;
    }
    if (agg_expr->get_arg()) {
      auto expr_range_info = getExprRangeInfo(agg_expr->get_arg());
      if ((expr_range_info.hash_type_ != GroupByColRangeType::OneColKnownRange || expr_range_info.has_nulls) &&
          order_entry.is_desc == order_entry.nulls_first) {
        return false;
      }
    }
    const auto& target_ti = target_expr->get_type_info();
    CHECK(!target_ti.is_array());
    if (!target_ti.is_integer()) {
      return false;
    }
  }
  return true;
}

bool QueryMemoryDescriptor::usesGetGroupValueFast() const {
  return (hash_type == GroupByColRangeType::OneColKnownRange && !getSmallBufferSizeBytes());
}

bool QueryMemoryDescriptor::usesCachedContext() const {
  return allow_multifrag && (usesGetGroupValueFast() || hash_type == GroupByColRangeType::MultiColPerfectHash);
}

bool QueryMemoryDescriptor::threadsShareMemory() const {
  return sharing == GroupByMemSharing::Shared;
}

bool QueryMemoryDescriptor::blocksShareMemory() const {
  if (executor_->isCPUOnly() || render_output) {
    return true;
  }
  return usesCachedContext() && !sharedMemBytes(ExecutorDeviceType::GPU) && many_entries(max_val, min_val, bucket);
}

bool QueryMemoryDescriptor::lazyInitGroups(const ExecutorDeviceType device_type) const {
  return device_type == ExecutorDeviceType::GPU && !render_output && !getSmallBufferSizeQuad();
}

bool QueryMemoryDescriptor::interleavedBins(const ExecutorDeviceType device_type) const {
  return interleaved_bins_on_gpu && device_type == ExecutorDeviceType::GPU;
}

size_t QueryMemoryDescriptor::sharedMemBytes(const ExecutorDeviceType device_type) const {
  CHECK(device_type == ExecutorDeviceType::CPU || device_type == ExecutorDeviceType::GPU);
  if (device_type == ExecutorDeviceType::CPU) {
    return 0;
  }
  const size_t shared_mem_threshold{0};
  const size_t shared_mem_bytes{getBufferSizeBytes(ExecutorDeviceType::GPU)};
  if (!usesGetGroupValueFast() || shared_mem_bytes > shared_mem_threshold) {
    return 0;
  }
  return shared_mem_bytes;
}

bool QueryMemoryDescriptor::canOutputColumnar() const {
  return usesGetGroupValueFast() && threadsShareMemory() && blocksShareMemory() &&
         !interleavedBins(ExecutorDeviceType::GPU);
}

bool QueryMemoryDescriptor::sortOnGpu() const {
  return sort_on_gpu_;
}

GroupByAndAggregate::DiamondCodegen::DiamondCodegen(llvm::Value* cond,
                                                    Executor* executor,
                                                    const bool chain_to_next,
                                                    const std::string& label_prefix,
                                                    DiamondCodegen* parent)
    : executor_(executor), chain_to_next_(chain_to_next), parent_(parent) {
  if (parent_) {
    CHECK(!chain_to_next_);
  }
  cond_true_ = llvm::BasicBlock::Create(LL_CONTEXT, label_prefix + "_true", ROW_FUNC);
  orig_cond_false_ = cond_false_ = llvm::BasicBlock::Create(LL_CONTEXT, label_prefix + "_false", ROW_FUNC);

  LL_BUILDER.CreateCondBr(cond, cond_true_, cond_false_);
  LL_BUILDER.SetInsertPoint(cond_true_);
}

void GroupByAndAggregate::DiamondCodegen::setChainToNext() {
  CHECK(!parent_);
  chain_to_next_ = true;
}

void GroupByAndAggregate::DiamondCodegen::setFalseTarget(llvm::BasicBlock* cond_false) {
  cond_false_ = cond_false;
}

GroupByAndAggregate::DiamondCodegen::~DiamondCodegen() {
  if (parent_) {
    LL_BUILDER.CreateBr(parent_->cond_false_);
  } else if (chain_to_next_) {
    LL_BUILDER.CreateBr(cond_false_);
  }
  LL_BUILDER.SetInsertPoint(orig_cond_false_);
}

void GroupByAndAggregate::patchGroupbyCall(llvm::CallInst* call_site) {
  CHECK(call_site);
  const auto func = call_site->getCalledFunction();
  const auto func_name = func->getName();
  if (func_name == "get_columnar_group_bin_offset") {
    return;
  }

  const auto arg_count = call_site->getNumArgOperands();
  const int32_t new_size_quad = query_mem_desc_.getRowSize() / sizeof(int64_t);
  std::vector<llvm::Value*> args;
  size_t arg_idx = 0;
  auto arg_iter = func->arg_begin();
  if (func_name == "get_group_value_one_key") {
    // param 7
    for (arg_idx = 0; arg_idx < 6; ++arg_idx, ++arg_iter) {
      args.push_back(call_site->getArgOperand(arg_idx));
    }
  } else {
    // param 5
    for (arg_idx = 0; arg_idx < 4; ++arg_idx, ++arg_iter) {
      args.push_back(call_site->getArgOperand(arg_idx));
    }
  }
  CHECK(arg_iter->getName() == "row_size_quad");
  CHECK_LT(arg_idx, arg_count);
  args.push_back(LL_INT(new_size_quad));
  ++arg_idx;
  for (; arg_idx < arg_count; ++arg_idx) {
    args.push_back(call_site->getArgOperand(arg_idx));
  }
  llvm::ReplaceInstWithInst(call_site, llvm::CallInst::Create(func, args));
}

bool GroupByAndAggregate::codegen(llvm::Value* filter_result, const CompilationOptions& co) {
  CHECK(filter_result);

  bool can_return_error = false;

  {
    const bool is_group_by = !ra_exe_unit_.groupby_exprs.empty();
    const auto query_mem_desc = getQueryMemoryDescriptor();

    DiamondCodegen filter_cfg(
        filter_result, executor_, !is_group_by || query_mem_desc.usesGetGroupValueFast(), "filter");

    if (is_group_by) {
      if (ra_exe_unit_.scan_limit) {
        auto crt_match_it = ROW_FUNC->arg_begin();
        ++crt_match_it;
        ++crt_match_it;
        LL_BUILDER.CreateStore(executor_->ll_int(int32_t(1)), crt_match_it);
      }

      auto agg_out_ptr_w_idx = codegenGroupBy(co, filter_cfg);
      if (query_mem_desc.usesGetGroupValueFast() ||
          query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
        if (query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
          filter_cfg.setChainToNext();
        }
        // Don't generate null checks if the group slot is guaranteed to be non-null,
        // as it's the case for get_group_value_fast* family.
        codegenAggCalls(agg_out_ptr_w_idx, {}, co);
      } else {
        {
          CHECK(!outputColumnar() || query_mem_desc.keyless_hash);
          DiamondCodegen nullcheck_cfg(LL_BUILDER.CreateICmpNE(std::get<0>(agg_out_ptr_w_idx),
                                                               llvm::ConstantPointerNull::get(llvm::PointerType::get(
                                                                   get_int_type(64, LL_CONTEXT), 0))),
                                       executor_,
                                       false,
                                       "groupby_nullcheck",
                                       &filter_cfg);
          codegenAggCalls(agg_out_ptr_w_idx, {}, co);
        }
        can_return_error = true;
        LL_BUILDER.CreateRet(LL_BUILDER.CreateNeg(LL_BUILDER.CreateTrunc(
            // TODO(alex): remove the trunc once pos is converted to 32 bits
            executor_->posArg(nullptr),
            get_int_type(32, LL_CONTEXT))));
      }

      if (!outputColumnar() && query_mem_desc.getRowSize() != query_mem_desc_.getRowSize()) {
        patchGroupbyCall(static_cast<llvm::CallInst*>(std::get<0>(agg_out_ptr_w_idx)));
      }

    } else {
      auto arg_it = ROW_FUNC->arg_begin();
      std::vector<llvm::Value*> agg_out_vec;
      for (int32_t i = 0; i < get_agg_count(ra_exe_unit_.target_exprs); ++i) {
        agg_out_vec.push_back(arg_it++);
      }
      codegenAggCalls(std::make_tuple(nullptr, nullptr), agg_out_vec, co);
    }
  }

  executor_->codegenInnerScanNextRow();

  return can_return_error;
}

std::tuple<llvm::Value*, llvm::Value*> GroupByAndAggregate::codegenGroupBy(const CompilationOptions& co,
                                                                           DiamondCodegen& diamond_codegen) {
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;

  const int32_t row_size_quad = outputColumnar() ? 0 : query_mem_desc_.getRowSize() / sizeof(int64_t);

  std::stack<llvm::BasicBlock*> array_loops;

  switch (query_mem_desc_.hash_type) {
    case GroupByColRangeType::OneColKnownRange:
    case GroupByColRangeType::OneColGuessedRange:
    case GroupByColRangeType::Scan: {
      CHECK_EQ(size_t(1), ra_exe_unit_.groupby_exprs.size());
      const auto group_expr = ra_exe_unit_.groupby_exprs.front();
      const auto group_expr_lv = executor_->groupByColumnCodegen(
          group_expr.get(),
          co,
          query_mem_desc_.has_nulls,
          query_mem_desc_.max_val + (query_mem_desc_.bucket ? query_mem_desc_.bucket : 1),
          diamond_codegen,
          array_loops);
      auto small_groups_buffer = arg_it;
      if (query_mem_desc_.usesGetGroupValueFast()) {
        std::string get_group_fn_name{outputColumnar() && !query_mem_desc_.keyless_hash
                                          ? "get_columnar_group_bin_offset"
                                          : "get_group_value_fast"};
        if (query_mem_desc_.keyless_hash) {
          get_group_fn_name += "_keyless";
        }
        if (query_mem_desc_.interleavedBins(co.device_type_)) {
          CHECK(!outputColumnar());
          CHECK(query_mem_desc_.keyless_hash);
          get_group_fn_name += "_semiprivate";
        }
        std::vector<llvm::Value*> get_group_fn_args{
            groups_buffer, group_expr_lv, LL_INT(query_mem_desc_.min_val), LL_INT(query_mem_desc_.bucket)};
        if (!query_mem_desc_.keyless_hash) {
          if (!outputColumnar()) {
            get_group_fn_args.push_back(LL_INT(row_size_quad));
          }
        } else {
          CHECK(!outputColumnar());
          get_group_fn_args.push_back(LL_INT(row_size_quad));
          if (query_mem_desc_.interleavedBins(co.device_type_)) {
            auto warp_idx = emitCall("thread_warp_idx", {LL_INT(executor_->warpSize())});
            get_group_fn_args.push_back(warp_idx);
            get_group_fn_args.push_back(LL_INT(executor_->warpSize()));
          }
        }
        if (get_group_fn_name == "get_columnar_group_bin_offset") {
          return std::make_tuple(groups_buffer, emitCall(get_group_fn_name, get_group_fn_args));
        }
        return std::make_tuple(emitCall(get_group_fn_name, get_group_fn_args), nullptr);
      } else {
        ++arg_it;
        return std::make_tuple(emitCall("get_group_value_one_key",
                                        {groups_buffer,
                                         LL_INT(static_cast<int32_t>(query_mem_desc_.entry_count)),
                                         small_groups_buffer,
                                         LL_INT(static_cast<int32_t>(query_mem_desc_.entry_count_small)),
                                         group_expr_lv,
                                         LL_INT(query_mem_desc_.min_val),
                                         LL_INT(row_size_quad),
                                         ++arg_it}),
                               nullptr);
      }
      break;
    }
    case GroupByColRangeType::MultiCol:
    case GroupByColRangeType::MultiColPerfectHash: {
      auto key_size_lv = LL_INT(static_cast<int32_t>(query_mem_desc_.group_col_widths.size()));
      // create the key buffer
      auto group_key = LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
      int32_t subkey_idx = 0;
      for (const auto group_expr : ra_exe_unit_.groupby_exprs) {
        auto col_range_info = getExprRangeInfo(group_expr.get());
        const auto group_expr_lv = executor_->groupByColumnCodegen(
            group_expr.get(), co, col_range_info.has_nulls, col_range_info.max + 1, diamond_codegen, array_loops);
        // store the sub-key to the buffer
        LL_BUILDER.CreateStore(group_expr_lv, LL_BUILDER.CreateGEP(group_key, LL_INT(subkey_idx++)));
      }
      ++arg_it;
      auto perfect_hash_func = query_mem_desc_.hash_type == GroupByColRangeType::MultiColPerfectHash
                                   ? codegenPerfectHashFunction()
                                   : nullptr;
      if (perfect_hash_func) {
        auto hash_lv = LL_BUILDER.CreateCall(perfect_hash_func, std::vector<llvm::Value*>{group_key});
        return std::make_tuple(emitCall("get_matching_group_value_perfect_hash",
                                        {groups_buffer, hash_lv, group_key, key_size_lv, LL_INT(row_size_quad)}),
                               nullptr);
      }
      return std::make_tuple(emitCall("get_group_value",
                                      {groups_buffer,
                                       LL_INT(static_cast<int32_t>(query_mem_desc_.entry_count)),
                                       group_key,
                                       key_size_lv,
                                       LL_INT(row_size_quad),
                                       ++arg_it}),
                             nullptr);
      break;
    }
    default:
      CHECK(false);
      break;
  }

  CHECK(false);
  return std::make_tuple(nullptr, nullptr);
}

llvm::Function* GroupByAndAggregate::codegenPerfectHashFunction() {
  CHECK_GT(ra_exe_unit_.groupby_exprs.size(), size_t(1));
  auto ft = llvm::FunctionType::get(get_int_type(32, LL_CONTEXT),
                                    std::vector<llvm::Type*>{llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)},
                                    false);
  auto key_hash_func =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "perfect_key_hash", executor_->cgen_state_->module_);
  executor_->cgen_state_->helper_functions_.push_back(key_hash_func);
  key_hash_func->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);
  auto& key_buff_arg = key_hash_func->getArgumentList().front();
  llvm::Value* key_buff_lv = &key_buff_arg;
  auto bb = llvm::BasicBlock::Create(LL_CONTEXT, "entry", key_hash_func);
  llvm::IRBuilder<> key_hash_func_builder(bb);
  llvm::Value* hash_lv{llvm::ConstantInt::get(get_int_type(64, LL_CONTEXT), 0)};
  std::vector<int64_t> cardinalities;
  for (const auto groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto col_range_info = getExprRangeInfo(groupby_expr.get());
    CHECK(col_range_info.hash_type_ == GroupByColRangeType::OneColKnownRange);
    cardinalities.push_back(col_range_info.max - col_range_info.min + 1);
  }
  size_t dim_idx = 0;
  for (const auto groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto key_comp_lv = key_hash_func_builder.CreateLoad(key_hash_func_builder.CreateGEP(key_buff_lv, LL_INT(dim_idx)));
    auto col_range_info = getExprRangeInfo(groupby_expr.get());
    auto crt_term_lv = key_hash_func_builder.CreateSub(key_comp_lv, LL_INT(col_range_info.min));
    for (size_t prev_dim_idx = 0; prev_dim_idx < dim_idx; ++prev_dim_idx) {
      crt_term_lv = key_hash_func_builder.CreateMul(crt_term_lv, LL_INT(cardinalities[prev_dim_idx]));
    }
    hash_lv = key_hash_func_builder.CreateAdd(hash_lv, crt_term_lv);
    ++dim_idx;
  }
  key_hash_func_builder.CreateRet(key_hash_func_builder.CreateTrunc(hash_lv, get_int_type(32, LL_CONTEXT)));
  return key_hash_func;
}

namespace {

std::vector<std::string> agg_fn_base_names(const TargetInfo& target_info) {
  if (!target_info.is_agg) {
    if ((target_info.sql_type.is_string() && target_info.sql_type.get_compression() == kENCODING_NONE) ||
        target_info.sql_type.is_array()) {
      return {"agg_id", "agg_id"};
    }
    return {"agg_id"};
  }
  switch (target_info.agg_kind) {
    case kAVG:
      return {"agg_sum", "agg_count"};
    case kCOUNT:
      return {target_info.is_distinct ? "agg_count_distinct" : "agg_count"};
    case kMAX:
      return {"agg_max"};
    case kMIN:
      return {"agg_min"};
    case kSUM:
      return {"agg_sum"};
    default:
      CHECK(false);
  }
}

}  // namespace

llvm::Value* GroupByAndAggregate::convertNullIfAny(const SQLTypeInfo& arg_type,
                                                   const SQLTypeInfo& agg_type,
                                                   const size_t chosen_bytes,
                                                   llvm::Value* target) {
  bool need_conversion{false};
  llvm::Value* arg_null{nullptr};
  llvm::Value* agg_null{nullptr};
  llvm::Value* target_to_cast{target};
  if (arg_type.is_fp()) {
    arg_null = executor_->inlineFpNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->inlineFpNull(agg_type);
      if (!static_cast<llvm::ConstantFP*>(arg_null)
               ->isExactlyValue(static_cast<llvm::ConstantFP*>(agg_null)->getValueAPF())) {
        need_conversion = true;
      }
    } else {
      // TODO(miyu): invalid case for now
      CHECK(false);
    }
  } else {
    arg_null = executor_->inlineIntNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->inlineFpNull(agg_type);
      need_conversion = true;
      target_to_cast = executor_->castToFP(target);
    } else {
      agg_null = executor_->inlineIntNull(agg_type);
      if ((static_cast<llvm::ConstantInt*>(arg_null)->getBitWidth() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getBitWidth()) ||
          (static_cast<llvm::ConstantInt*>(arg_null)->getValue() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getValue())) {
        need_conversion = true;
      }
    }
  }
  if (need_conversion) {
    auto cmp =
        arg_type.is_fp() ? LL_BUILDER.CreateFCmpOEQ(target, arg_null) : LL_BUILDER.CreateICmpEQ(target, arg_null);
    return LL_BUILDER.CreateSelect(cmp, agg_null, executor_->castToTypeIn(target_to_cast, chosen_bytes << 3));
  } else {
    return target;
  }
}

void GroupByAndAggregate::codegenAggCalls(const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                                          const std::vector<llvm::Value*>& agg_out_vec,
                                          const CompilationOptions& co) {
  // TODO(alex): unify the two cases, the output for non-group by queries
  //             should be a contiguous buffer
  const bool is_group_by{std::get<0>(agg_out_ptr_w_idx)};
  if (is_group_by) {
    CHECK(agg_out_vec.empty());
  } else {
    CHECK(!agg_out_vec.empty());
  }
  int32_t agg_out_off{0};
  for (size_t target_idx = 0; target_idx < ra_exe_unit_.target_exprs.size(); ++target_idx) {
    auto target_expr = ra_exe_unit_.target_exprs[target_idx];
    CHECK(target_expr);
    if (dynamic_cast<Analyzer::UOper*>(target_expr) &&
        static_cast<Analyzer::UOper*>(target_expr)->get_optype() == kUNNEST) {
      throw std::runtime_error("UNNEST not supported in the projection list yet.");
    }
    auto agg_info = compact_target_info(target_expr);
    auto arg_expr = agg_arg(target_expr);
    if (arg_expr && constrained_not_null(arg_expr, ra_exe_unit_.quals)) {
      agg_info.skip_null_val = false;
    }
    const auto agg_fn_names = agg_fn_base_names(agg_info);
    auto target_lvs = codegenAggArg(target_expr, co);
    if (executor_->plan_state_->isLazyFetchColumn(target_expr) || !is_group_by) {
      // TODO(miyu): could be smaller than qword
      query_mem_desc_.agg_col_widths[agg_out_off].compact = sizeof(int64_t);
    }
    llvm::Value* str_target_lv{nullptr};
    if (target_lvs.size() == 3) {
      // none encoding string, pop the packed pointer + length since
      // it's only useful for IS NULL checks and assumed to be only
      // two components (pointer and length) for the purpose of projection
      str_target_lv = target_lvs.front();
      target_lvs.erase(target_lvs.begin());
    }
    if (target_lvs.size() < agg_fn_names.size()) {
      CHECK_EQ(size_t(1), target_lvs.size());
      CHECK_EQ(size_t(2), agg_fn_names.size());
      for (size_t i = 1; i < agg_fn_names.size(); ++i) {
        target_lvs.push_back(target_lvs.front());
      }
    } else {
      CHECK(str_target_lv || (agg_fn_names.size() == target_lvs.size()));
      CHECK(target_lvs.size() == 1 || target_lvs.size() == 2);
    }
    uint32_t col_off{0};
    const bool is_simple_count = agg_info.is_agg && agg_info.agg_kind == kCOUNT && !agg_info.is_distinct;
    if (co.device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory() && is_simple_count &&
        (!arg_expr || arg_expr->get_type_info().get_notnull())) {
      CHECK_EQ(size_t(1), agg_fn_names.size());
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[agg_out_off].compact;
      llvm::Value* agg_col_ptr{nullptr};
      if (is_group_by) {
        if (outputColumnar()) {
          col_off = query_mem_desc_.getColOffInBytes(0, agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          CHECK(std::get<1>(agg_out_ptr_w_idx));
          auto offset = LL_BUILDER.CreateAdd(std::get<1>(agg_out_ptr_w_idx), LL_INT(col_off));
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              offset);
        } else {
          col_off = query_mem_desc_.getColOnlyOffInBytes(agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              LL_INT(col_off));
        }
      }

      llvm::Value* acc_i32 = nullptr;
      if (chosen_bytes != sizeof(int32_t)) {
        acc_i32 = LL_BUILDER.CreateBitCast(is_group_by ? agg_col_ptr : agg_out_vec[agg_out_off],
                                           llvm::PointerType::get(get_int_type(32, LL_CONTEXT), 0));
      } else {
        acc_i32 = (is_group_by ? agg_col_ptr : agg_out_vec[agg_out_off]);
      }
      LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add, acc_i32, LL_INT(1), llvm::AtomicOrdering::Monotonic);
      ++agg_out_off;
      continue;
    }
    size_t target_lv_idx = 0;
    const bool lazy_fetched{executor_->plan_state_->isLazyFetchColumn(target_expr)};
    for (const auto& agg_base_name : agg_fn_names) {
      if (agg_info.agg_kind == kCOUNT && !agg_info.is_distinct && arg_expr) {
        agg_info.sql_type = arg_expr->get_type_info();
      }
      if (agg_info.is_distinct && arg_expr->get_type_info().is_array()) {
        CHECK(agg_info.is_distinct);
        CHECK_EQ(static_cast<size_t>(query_mem_desc_.agg_col_widths[agg_out_off].actual), sizeof(int64_t));
        // TODO(miyu): check if buffer may be columnar here
        CHECK(!outputColumnar());
        const auto& elem_ti = arg_expr->get_type_info().get_elem_type();
        if (is_group_by) {
          col_off = query_mem_desc_.getColOnlyOffInBytes(agg_out_off);
          CHECK_EQ(size_t(0), col_off % sizeof(int64_t));
          col_off /= sizeof(int64_t);
        }
        executor_->cgen_state_->emitExternalCall(
            "agg_count_distinct_array_" + numeric_type_name(elem_ti),
            llvm::Type::getVoidTy(LL_CONTEXT),
            {is_group_by ? LL_BUILDER.CreateGEP(std::get<0>(agg_out_ptr_w_idx), LL_INT(col_off))
                         : agg_out_vec[agg_out_off],
             target_lvs[target_lv_idx],
             executor_->posArg(arg_expr),
             elem_ti.is_fp() ? static_cast<llvm::Value*>(executor_->inlineFpNull(elem_ti))
                             : static_cast<llvm::Value*>(executor_->inlineIntNull(elem_ti))});
        ++agg_out_off;
        ++target_lv_idx;
        continue;
      }

      llvm::Value* agg_col_ptr{nullptr};
      const size_t chosen_bytes = static_cast<size_t>(query_mem_desc_.agg_col_widths[agg_out_off].compact);
      if (is_group_by) {
        if (outputColumnar()) {
          col_off = query_mem_desc_.getColOffInBytes(0, agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          CHECK(std::get<1>(agg_out_ptr_w_idx));
          auto offset = LL_BUILDER.CreateAdd(std::get<1>(agg_out_ptr_w_idx), LL_INT(col_off));
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              offset);
        } else {
          col_off = query_mem_desc_.getColOnlyOffInBytes(agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              LL_INT(col_off));
        }
      }

      auto target_lv = target_lvs[target_lv_idx];
      // TODO(miyu): check proper condition to choose skip_val version for non-groupby
      const bool need_skip_null =
          agg_info.skip_null_val && !(agg_info.agg_kind == kAVG && agg_base_name == "agg_count");
      if (need_skip_null && agg_info.agg_kind != kCOUNT) {
        target_lv = convertNullIfAny(arg_expr->get_type_info(), agg_info.sql_type, chosen_bytes, target_lv);
      } else if (!lazy_fetched && agg_info.sql_type.is_fp()) {
        target_lv = executor_->castToFP(target_lv);
      }

      target_lv = executor_->castToTypeIn(target_lv, (chosen_bytes << 3));

      std::vector<llvm::Value*> agg_args{
          is_group_by ? agg_col_ptr : executor_->castToIntPtrTyIn(agg_out_vec[agg_out_off], (chosen_bytes << 3)),
          (is_simple_count && !arg_expr) ? (chosen_bytes == sizeof(int32_t) ? LL_INT(int32_t(0)) : LL_INT(int64_t(0)))
                                         : (is_simple_count && arg_expr && str_target_lv ? str_target_lv : target_lv)};
      std::string agg_fname{agg_base_name};
      if (!lazy_fetched && agg_info.sql_type.is_fp()) {
        if (!lazy_fetched) {
          if (chosen_bytes == sizeof(float)) {
            CHECK_EQ(agg_info.sql_type.get_type(), kFLOAT);
            agg_fname += "_float";
          } else {
            CHECK_EQ(chosen_bytes, sizeof(double));
            agg_fname += "_double";
          }
        }
      } else if (chosen_bytes == sizeof(int32_t)) {
        agg_fname += "_int32";
      }

      if (agg_info.is_distinct) {
        CHECK_EQ(chosen_bytes, sizeof(int64_t));
        CHECK(!agg_info.sql_type.is_fp());
        CHECK_EQ("agg_count_distinct", agg_base_name);
        codegenCountDistinct(target_idx, target_expr, agg_args, query_mem_desc_, co.device_type_);
      } else {
        if (need_skip_null) {
          agg_fname += "_skip_val";
          auto null_lv = executor_->castToTypeIn(
              agg_info.sql_type.is_fp() ? static_cast<llvm::Value*>(executor_->inlineFpNull(agg_info.sql_type))
                                        : static_cast<llvm::Value*>(executor_->inlineIntNull(agg_info.sql_type)),
              (chosen_bytes << 3));
          agg_args.push_back(null_lv);
        }
        if (!agg_info.is_distinct) {
          emitCall((co.device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory())
                       ? agg_fname + "_shared"
                       : agg_fname,
                   agg_args);
        }
      }
      ++agg_out_off;
      ++target_lv_idx;
    }
  }
  for (auto target_expr : ra_exe_unit_.target_exprs) {
    CHECK(target_expr);
    executor_->plan_state_->isLazyFetchColumn(target_expr);
  }
}

void GroupByAndAggregate::codegenCountDistinct(const size_t target_idx,
                                               const Analyzer::Expr* target_expr,
                                               std::vector<llvm::Value*>& agg_args,
                                               const QueryMemoryDescriptor& query_mem_desc,
                                               const ExecutorDeviceType device_type) {
  const auto agg_info = compact_target_info(target_expr);
  const auto& arg_ti = static_cast<const Analyzer::AggExpr*>(target_expr)->get_arg()->get_type_info();
  if (arg_ti.is_fp()) {
    agg_args.back() = executor_->cgen_state_->ir_builder_.CreateBitCast(
        agg_args.back(), get_int_type(64, executor_->cgen_state_->context_));
  }
  CHECK(device_type == ExecutorDeviceType::CPU);
  auto it_count_distinct = query_mem_desc.count_distinct_descriptors_.find(target_idx);
  CHECK(it_count_distinct != query_mem_desc.count_distinct_descriptors_.end());
  std::string agg_fname{"agg_count_distinct"};
  if (it_count_distinct->second.impl_type_ == CountDistinctImplType::Bitmap) {
    agg_fname += "_bitmap";
    agg_args.push_back(LL_INT(static_cast<int64_t>(it_count_distinct->second.min_val)));
  }
  if (agg_info.skip_null_val) {
    auto null_lv =
        executor_->castToTypeIn((arg_ti.is_fp() ? static_cast<llvm::Value*>(executor_->inlineFpNull(arg_ti))
                                                : static_cast<llvm::Value*>(executor_->inlineIntNull(arg_ti))),
                                64);
    null_lv =
        executor_->cgen_state_->ir_builder_.CreateBitCast(null_lv, get_int_type(64, executor_->cgen_state_->context_));
    agg_fname += "_skip_val";
    agg_args.push_back(null_lv);
  }
  if (it_count_distinct->second.impl_type_ == CountDistinctImplType::Bitmap) {
    emitCall(agg_fname, agg_args);
  } else {
    emitCall(agg_fname, agg_args);
  }
}

std::vector<llvm::Value*> GroupByAndAggregate::codegenAggArg(const Analyzer::Expr* target_expr,
                                                             const CompilationOptions& co) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  // TODO(alex): handle arrays uniformly?
  if (target_expr) {
    const auto& target_ti = target_expr->get_type_info();
    if (target_ti.is_array() && !executor_->plan_state_->isLazyFetchColumn(target_expr)) {
      const auto target_lvs = executor_->codegen(target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
      CHECK_EQ(size_t(1), target_lvs.size());
      CHECK(!agg_expr);
      const auto i32_ty = get_int_type(32, executor_->cgen_state_->context_);
      const auto i8p_ty = llvm::PointerType::get(get_int_type(8, executor_->cgen_state_->context_), 0);
      const auto& elem_ti = target_ti.get_elem_type();
      return {
          executor_->cgen_state_->emitExternalCall(
              "array_buff", i8p_ty, {target_lvs.front(), executor_->posArg(target_expr)}),
          executor_->cgen_state_->emitExternalCall(
              "array_size",
              i32_ty,
              {target_lvs.front(), executor_->posArg(target_expr), executor_->ll_int(log2_bytes(elem_ti.get_size()))})};
    }
  }
  return agg_expr ? executor_->codegen(agg_expr->get_arg(), true, co)
                  : executor_->codegen(target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
}

llvm::Value* GroupByAndAggregate::emitCall(const std::string& fname, const std::vector<llvm::Value*>& args) {
  return executor_->cgen_state_->emitCall(fname, args);
}

#undef ROW_FUNC
#undef LL_INT
#undef LL_BUILDER
#undef LL_CONTEXT
