#include "GroupByAndAggregate.h"
#include "ExpressionRange.h"

#include "Execute.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "../CudaMgr/CudaMgr.h"
#include "../Utils/ChunkIter.h"

#include <numeric>
#include <thread>


void ResultRows::addKeylessGroupByBuffer(const int64_t* group_by_buffer,
                                         const int32_t groups_buffer_entry_count,
                                         const int64_t min_val,
                                         const int8_t warp_count) {
  const size_t agg_col_count { targets_.size() };
  std::vector<int64_t> partial_agg_vals(agg_col_count, 0);
  std::vector<int64_t> agg_vals(agg_col_count, 0);
  simple_keys_.reserve(groups_buffer_entry_count);
  target_values_.reserve(groups_buffer_entry_count);
  for (int32_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
    memset(&partial_agg_vals[0], 0, agg_col_count * sizeof(partial_agg_vals[0]));
    memset(&agg_vals[0], 0, agg_col_count * sizeof(agg_vals[0]));
    beginRow(bin + min_val);
    bool discard_row = true;
    size_t group_by_buffer_base_idx { warp_count * bin * agg_col_count };
    for (int8_t warp_idx = 0; warp_idx < warp_count; ++warp_idx) {
      bool discard_partial_result = true;
      for (size_t target_idx = 0; target_idx < agg_col_count; ++target_idx) {
        const auto& agg_info = targets_[target_idx];
        CHECK(!agg_info.is_agg || (agg_info.is_agg && agg_info.agg_kind == kCOUNT));
        partial_agg_vals[target_idx] = group_by_buffer[group_by_buffer_base_idx + target_idx];
        auto partial_bin_val = partial_agg_vals[target_idx];
        if (agg_info.is_distinct) {
          CHECK(agg_info.is_agg && agg_info.agg_kind == kCOUNT);
          partial_bin_val = bitmap_set_size(partial_bin_val, target_idx, row_set_mem_owner_->count_distinct_descriptors_);
        }
        if (agg_info.is_agg && partial_bin_val) {
          discard_partial_result = false;
        }
      }
      group_by_buffer_base_idx += agg_col_count;
      if (discard_partial_result) {
        continue;
      }
      discard_row = false;
      for (size_t target_idx = 0; target_idx < agg_col_count; ++target_idx) {
        const auto& agg_info = targets_[target_idx];
        auto partial_bin_val = partial_agg_vals[target_idx];
        if (agg_info.is_agg) {
          CHECK_EQ(kCOUNT, agg_info.agg_kind);
          agg_vals[target_idx] += partial_bin_val;
        } else {
          if (agg_vals[target_idx]) {
            CHECK_EQ(agg_vals[target_idx], partial_bin_val);
          } else {
            agg_vals[target_idx] = partial_bin_val;
          }
        }
      }
    }
    if (discard_row) {
      discardRow();
      continue;
    }
    addValues(agg_vals);
  }
}

void ResultRows::reduce(const ResultRows& other_results) {
  if (other_results.empty()) {
    return;
  }
  if (empty()) {
    *this = other_results;
    return;
  }

  if (group_by_buffer_) {
    CHECK(other_results.group_by_buffer_);
    const size_t agg_col_count { targets_.size() };
    for (size_t target_idx = 0; target_idx < agg_col_count; ++target_idx) {
      const auto& agg_info = targets_[target_idx];
      CHECK(!agg_info.is_agg || (agg_info.is_agg && agg_info.agg_kind == kCOUNT));
      CountDistinctImplType count_distinct_type { CountDistinctImplType::Bitmap };
      size_t bitmap_sz_bytes { 0 };
      if (agg_info.is_distinct) {
        CHECK(agg_info.is_agg);
        CHECK_EQ(kCOUNT, agg_info.agg_kind);
        auto count_distinct_desc_it = row_set_mem_owner_->count_distinct_descriptors_.find(target_idx);
        CHECK(count_distinct_desc_it != row_set_mem_owner_->count_distinct_descriptors_.end());
        count_distinct_type = count_distinct_desc_it->second.impl_type_;
        bitmap_sz_bytes = count_distinct_desc_it->second.bitmapSizeBytes();
      }
      for (int32_t bin = 0; bin < groups_buffer_entry_count_; ++bin) {
        size_t group_by_buffer_base_idx { bin * warp_count_ * agg_col_count + target_idx };
        for (int8_t warp_idx = 0; warp_idx < warp_count_; ++warp_idx) {
          const auto val = other_results.group_by_buffer_[group_by_buffer_base_idx];
          if (agg_info.is_agg) {
            if (agg_info.is_distinct) {
              if (count_distinct_type == CountDistinctImplType::Bitmap) {
                auto old_set = reinterpret_cast<int8_t*>(group_by_buffer_[group_by_buffer_base_idx]);
                auto new_set = reinterpret_cast<int8_t*>(val);
                bitmap_set_unify(new_set, old_set, bitmap_sz_bytes);
              } else {
                CHECK(count_distinct_type == CountDistinctImplType::StdSet);
                auto old_set = reinterpret_cast<std::set<int64_t>*>(group_by_buffer_[group_by_buffer_base_idx]);
                auto new_set = reinterpret_cast<std::set<int64_t>*>(val);
                old_set->insert(new_set->begin(), new_set->end());
                new_set->insert(old_set->begin(), old_set->end());
              }
            } else {
              group_by_buffer_[group_by_buffer_base_idx] += val;
            }
          } else {
            if (val) {
              group_by_buffer_[group_by_buffer_base_idx] = val;
            }
          }
          group_by_buffer_base_idx += agg_col_count;
        }
      }
    }
    return;
  }

  auto reduce_impl = [this](InternalTargetValue* crt_val, const InternalTargetValue* new_val,
      const TargetInfo& agg_info, const size_t agg_col_idx) {
    CHECK(agg_info.sql_type.is_integer() || agg_info.sql_type.is_time() ||
          agg_info.sql_type.get_type() == kBOOLEAN || agg_info.sql_type.is_string() ||
          agg_info.sql_type.get_type() == kFLOAT || agg_info.sql_type.get_type() == kDOUBLE);
    switch (agg_info.agg_kind) {
    case kSUM:
    case kCOUNT:
    case kAVG:
      if (agg_info.is_distinct) {
        CHECK(agg_info.is_agg);
        CHECK_EQ(kCOUNT, agg_info.agg_kind);
        auto count_distinct_desc_it = row_set_mem_owner_->count_distinct_descriptors_.find(agg_col_idx);
        CHECK(count_distinct_desc_it != row_set_mem_owner_->count_distinct_descriptors_.end());
        if (count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::Bitmap) {
          auto old_set = reinterpret_cast<int8_t*>(crt_val->i1);
          auto new_set = reinterpret_cast<int8_t*>(new_val->i1);
          bitmap_set_unify(new_set, old_set, count_distinct_desc_it->second.bitmapSizeBytes());
        } else {
          CHECK(count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::StdSet);
          auto old_set = reinterpret_cast<std::set<int64_t>*>(crt_val->i1);
          auto new_set = reinterpret_cast<std::set<int64_t>*>(new_val->i1);
          old_set->insert(new_set->begin(), new_set->end());
          new_set->insert(old_set->begin(), old_set->end());
        }
        break;
      }
      if (agg_info.agg_kind == kAVG) {
        CHECK(crt_val->isPair());
        CHECK(new_val->isPair());
        if (agg_info.sql_type.is_fp()) {
          agg_sum_double(
            &crt_val->i1,
            *reinterpret_cast<const double*>(&new_val->i1));
        } else {
          agg_sum(&crt_val->i1, new_val->i1);
        }
        agg_sum(&crt_val->i2, new_val->i2);
        break;
      }
      if (agg_info.sql_type.is_integer() || agg_info.sql_type.is_time()) {
        agg_sum(&crt_val->i1, new_val->i1);
      } else {
        agg_sum_double(&crt_val->i1, *reinterpret_cast<const double*>(&new_val->i1));
      }
      break;
    case kMIN:
      if (agg_info.sql_type.is_integer() || agg_info.sql_type.is_time() || agg_info.sql_type.is_boolean()) {
        if (agg_info.skip_null_val) {
          agg_min_skip_val(&crt_val->i1, new_val->i1, inline_int_null_val(agg_info.sql_type));
        } else {
          agg_min(&crt_val->i1, new_val->i1);
        }
      } else {
        if (agg_info.skip_null_val) {
          agg_min_double_skip_val(&crt_val->i1, *reinterpret_cast<const double*>(&new_val->i1),
            inline_fp_null_val(agg_info.sql_type));
        } else {
          agg_min_double(&crt_val->i1, *reinterpret_cast<const double*>(&new_val->i1));
        }
      }
      break;
    case kMAX:
      if (agg_info.sql_type.is_integer() || agg_info.sql_type.is_time()) {
        if (agg_info.skip_null_val) {
          agg_max_skip_val(&crt_val->i1, new_val->i1, inline_int_null_val(agg_info.sql_type));
        } else {
          agg_max(&crt_val->i1, new_val->i1);
        }
      } else {
        if (agg_info.skip_null_val) {
          agg_max_double_skip_val(&crt_val->i1, *reinterpret_cast<const double*>(&new_val->i1),
            inline_fp_null_val(agg_info.sql_type));
        } else {
          agg_max_double(&crt_val->i1, *reinterpret_cast<const double*>(&new_val->i1));
        }
      }
      break;
    default:
      CHECK(false);
    }
  };

  if (simple_keys_.empty() && multi_keys_.empty()) {
    CHECK_EQ(1, size());
    CHECK_EQ(1, other_results.size());
    auto& crt_results = target_values_.front();
    const auto& new_results = other_results.target_values_.front();
    for (size_t agg_col_idx = 0; agg_col_idx < colCount(); ++agg_col_idx) {
      const auto agg_info = targets_[agg_col_idx];
      reduce_impl(&crt_results[agg_col_idx], &new_results[agg_col_idx],
        agg_info, agg_col_idx);
    }
    return;
  }

  CHECK_NE(simple_keys_.empty(), multi_keys_.empty());
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
      const auto agg_info = targets_[agg_col_idx];
      reduce_impl(&old_agg_results[agg_col_idx], &kv.second[agg_col_idx], agg_info, agg_col_idx);
    }
  }
  for (const auto& kv : other_results.as_unordered_map_) {
    auto it = as_unordered_map_.find(kv.first);
    if (it == as_unordered_map_.end()) {
      as_unordered_map_.insert(std::make_pair(kv.first, kv.second));
      continue;
    }
    auto& old_agg_results = it->second;
    CHECK_EQ(old_agg_results.size(), kv.second.size());
    const size_t agg_col_count = old_agg_results.size();
    for (size_t agg_col_idx = 0; agg_col_idx < agg_col_count; ++agg_col_idx) {
      const auto agg_info = targets_[agg_col_idx];
      reduce_impl(&old_agg_results[agg_col_idx], &kv.second[agg_col_idx], agg_info, agg_col_idx);
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

__attribute__((always_inline))
inline double pair_to_double(const std::pair<int64_t, int64_t>& fp_pair, const bool is_int) {
  return is_int
    ? static_cast<double>(fp_pair.first) / static_cast<double>(fp_pair.second)
    : *reinterpret_cast<const double*>(&fp_pair.first) / static_cast<double>(fp_pair.second);
}

}  // namespace

#define LIKELY(x)       __builtin_expect((x),1)
#define UNLIKELY(x)     __builtin_expect((x),0)

void ResultRows::sort(const Planner::Sort* sort_plan, const int64_t top_n) {
  const auto& target_list = sort_plan->get_targetlist();
  const auto& order_entries = sort_plan->get_order_entries();
  const bool use_heap { order_entries.size() == 1 && !sort_plan->get_remove_duplicates() && top_n };
  // TODO(alex): check the semantics for order by multiple columns
  for (const auto order_entry : boost::adaptors::reverse(order_entries)) {
    CHECK_GE(order_entry.tle_no, 1);
    CHECK_LE(order_entry.tle_no, target_list.size());
    auto compare = [this, &order_entry, use_heap](const TargetValues& lhs, const TargetValues& rhs) {
      // The compare function must define a strict weak ordering, which means
      // we can't use the overloaded less than operator for boost::variant since
      // there's not greater than counterpart. If we naively use "not less than"
      // as the compare function for descending order, std::sort will trigger
      // a segmentation fault (or corrupt memory).
      const auto& entry_ti = targets_[order_entry.tle_no - 1].sql_type;
      const auto is_int = entry_ti.is_integer();
      const auto is_dict = entry_ti.is_string() &&
        entry_ti.get_compression() == kENCODING_DICT;
      const auto& lhs_v = lhs[order_entry.tle_no - 1];
      const auto& rhs_v = rhs[order_entry.tle_no - 1];
      if (UNLIKELY(isNull(entry_ti, lhs_v) && isNull(entry_ti, rhs_v))) {
        return false;
      }
      if (UNLIKELY(isNull(entry_ti, lhs_v) && !isNull(entry_ti, rhs_v))) {
        return order_entry.nulls_first;
      }
      if (UNLIKELY(isNull(entry_ti, rhs_v) && !isNull(entry_ti, lhs_v))) {
        return !order_entry.nulls_first;
      }
      const bool use_desc_cmp = use_heap ? !order_entry.is_desc : order_entry.is_desc;
      if (LIKELY(lhs_v.isInt())) {
        CHECK(rhs_v.isInt());
        if (UNLIKELY(is_dict)) {
          CHECK_EQ(4, entry_ti.get_size());
          auto string_dict = executor_->getStringDictionary(entry_ti.get_comp_param(), row_set_mem_owner_);
          auto lhs_str = string_dict->getString(lhs_v.i1);
          auto rhs_str = string_dict->getString(rhs_v.i1);
          return use_desc_cmp ? lhs_str > rhs_str : lhs_str < rhs_str;
        }
        if (UNLIKELY(targets_[order_entry.tle_no - 1].is_distinct)) {
          const auto lhs_sz = bitmap_set_size(lhs_v.i1, order_entry.tle_no - 1,
            row_set_mem_owner_->count_distinct_descriptors_);
          const auto rhs_sz = bitmap_set_size(rhs_v.i1, order_entry.tle_no - 1,
            row_set_mem_owner_->count_distinct_descriptors_);
          return use_desc_cmp ? lhs_sz > rhs_sz : lhs_sz < rhs_sz;
        }
        return use_desc_cmp ? lhs_v.i1 > rhs_v.i1 : lhs_v.i1 < rhs_v.i1;
      } else {
        if (lhs_v.isPair()) {
          CHECK(rhs_v.isPair());
          return use_desc_cmp
            ? pair_to_double({ lhs_v.i1, lhs_v.i2 }, is_int) > pair_to_double({ rhs_v.i1, rhs_v.i2 }, is_int)
            : pair_to_double({ lhs_v.i1, lhs_v.i2 }, is_int) < pair_to_double({ rhs_v.i1, rhs_v.i2 }, is_int);
        } else {
          CHECK(lhs_v.isStr() && rhs_v.isStr());
          return use_desc_cmp ? lhs_v.strVal() > rhs_v.strVal() : lhs_v.strVal() < rhs_v.strVal();
        }
      }
    };
    if (use_heap) {
      std::make_heap(target_values_.begin(), target_values_.end(), compare);
      decltype(target_values_) top_target_values;
      top_target_values.reserve(top_n);
      for (int64_t i = 0; i < top_n && !target_values_.empty(); ++i) {
        top_target_values.push_back(target_values_.front());
        std::pop_heap(target_values_.begin(), target_values_.end(), compare);
        target_values_.pop_back();
      }
      target_values_.swap(top_target_values);
      return;
    }
    std::sort(target_values_.begin(), target_values_.end(), compare);
  }
  if (sort_plan->get_remove_duplicates()) {
    std::sort(target_values_.begin(), target_values_.end());
    target_values_.erase(std::unique(target_values_.begin(), target_values_.end()), target_values_.end());
  }
}

#undef UNLIKELY
#undef LIKELY

TargetValue ResultRows::get(const size_t row_idx,
                            const size_t col_idx,
                            const bool translate_strings) const {
  CHECK_GE(row_idx, 0);
  CHECK_LT(row_idx, target_values_.size());
  CHECK_GE(col_idx, 0);
  CHECK_LT(col_idx, targets_.size());
  const auto agg_info = targets_[col_idx];
  if (agg_info.agg_kind == kAVG) {
    CHECK(!targets_[col_idx].sql_type.is_string());
    const auto& row_vals = target_values_[row_idx];
    CHECK_LT(col_idx, row_vals.size());
    CHECK(row_vals[col_idx].isPair());
    return pair_to_double({ row_vals[col_idx].i1, row_vals[col_idx].i2 }, targets_[col_idx].sql_type.is_integer());
  }
  if (targets_[col_idx].sql_type.is_integer() ||
      targets_[col_idx].sql_type.is_boolean() ||
      targets_[col_idx].sql_type.is_time()) {
    if (agg_info.is_distinct) {
      return TargetValue(bitmap_set_size(target_values_[row_idx][col_idx].i1, col_idx,
        row_set_mem_owner_->count_distinct_descriptors_));
    }
    CHECK_LT(col_idx, target_values_[row_idx].size());
    const auto v = target_values_[row_idx][col_idx];
    CHECK(v.isInt());
    return v.i1;
  } else if (targets_[col_idx].sql_type.is_string()) {
    if (targets_[col_idx].sql_type.get_compression() == kENCODING_DICT) {
      const int dict_id = targets_[col_idx].sql_type.get_comp_param();
      const auto string_id = target_values_[row_idx][col_idx].i1;
      if (!translate_strings) {
        return TargetValue(string_id);
      }
      return string_id == NULL_INT
        ? TargetValue(nullptr)
        : TargetValue(executor_->getStringDictionary(dict_id, row_set_mem_owner_)->getString(string_id));
    } else {
      CHECK_EQ(kENCODING_NONE, targets_[col_idx].sql_type.get_compression());
      return target_values_[row_idx][col_idx].isNull()
        ? TargetValue(nullptr)
        : TargetValue(target_values_[row_idx][col_idx].strVal());
    }
  } else {
    CHECK(targets_[col_idx].sql_type.is_fp());
    return TargetValue(*reinterpret_cast<const double*>(&target_values_[row_idx][col_idx].i1));
  }
  CHECK(false);
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

QueryExecutionContext::QueryExecutionContext(
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const int device_id,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
  : query_mem_desc_(query_mem_desc)
  , init_agg_vals_(executor->plan_state_->init_agg_vals_)
  , executor_(executor)
  , device_type_(device_type)
  , device_id_(device_id)
  , col_buffers_(col_buffers)
  , num_buffers_ { device_type == ExecutorDeviceType::CPU
      ? 1
      : executor->blockSize() * executor->gridSize() }
  , row_set_mem_owner_(row_set_mem_owner) {
  if (query_mem_desc_.group_col_widths.empty()) {
    allocateCountDistinctBuffers(false);
    return;
  }

  std::vector<int64_t> group_by_buffer_template(query_mem_desc_.getBufferSizeQuad(device_type));
  if (!query_mem_desc_.lazyInitGroups(device_type)) {
    initGroups(&group_by_buffer_template[0], &init_agg_vals[0],
      query_mem_desc_.entry_count, query_mem_desc_.keyless_hash,
      query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1);
  }

  if (query_mem_desc_.interleavedBins(device_type_)) {
    CHECK(query_mem_desc_.keyless_hash);
  }

  if (query_mem_desc_.keyless_hash) {
    CHECK_EQ(0, query_mem_desc_.getSmallBufferSizeQuad());
  }

  std::vector<int64_t> group_by_small_buffer_template;
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    group_by_small_buffer_template.resize(query_mem_desc_.getSmallBufferSizeQuad());
    initGroups(&group_by_small_buffer_template[0], &init_agg_vals[0],
      query_mem_desc_.entry_count_small, false, 1);
  }

  size_t step { device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory()
    ? executor_->blockSize() : 1 };

  for (size_t i = 0; i < num_buffers_; i += step) {
    auto group_by_buffer = static_cast<int64_t*>(malloc(query_mem_desc_.getBufferSizeBytes(device_type_)));
    if (!query_mem_desc_.lazyInitGroups(device_type)) {
      memcpy(group_by_buffer, &group_by_buffer_template[0], query_mem_desc_.getBufferSizeBytes(device_type_));
    }
    row_set_mem_owner_->addGroupByBuffer(group_by_buffer);
    group_by_buffers_.push_back(group_by_buffer);
    for (size_t j = 1; j < step; ++j) {
      group_by_buffers_.push_back(nullptr);
    }
    if (query_mem_desc_.getSmallBufferSizeBytes()) {
      auto group_by_small_buffer = static_cast<int64_t*>(malloc(query_mem_desc_.getSmallBufferSizeBytes()));
      row_set_mem_owner_->addGroupByBuffer(group_by_small_buffer);
      memcpy(group_by_small_buffer, &group_by_small_buffer_template[0], query_mem_desc_.getSmallBufferSizeBytes());
      small_group_by_buffers_.push_back(group_by_small_buffer);
      for (size_t j = 1; j < step; ++j) {
        small_group_by_buffers_.push_back(nullptr);
      }
    }
  }
}

void QueryExecutionContext::initGroups(int64_t* groups_buffer,
                                       const int64_t* init_vals,
                                       const int32_t groups_buffer_entry_count,
                                       const bool keyless,
                                       const size_t warp_size) {
  auto agg_bitmap_size = allocateCountDistinctBuffers(true);
  const int32_t agg_col_count = query_mem_desc_.agg_col_widths.size();
  const int32_t key_qw_count = query_mem_desc_.group_col_widths.size();
  if (keyless) {
    assert(warp_size >= 1);
    assert(key_qw_count == 1);
    for (int32_t i = 0; i < groups_buffer_entry_count * agg_col_count * static_cast<int32_t>(warp_size); ++i) {
      auto init_idx = i % agg_col_count;
      const ssize_t bitmap_sz { agg_bitmap_size[init_idx] };
      if (!bitmap_sz) {
        groups_buffer[i] = init_vals[init_idx];
      } else {
        groups_buffer[i] = bitmap_sz > 0 ? allocateCountDistinctBitmap(bitmap_sz) : allocateCountDistinctSet();
      }
    }
    return;
  }
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + agg_col_count);
  for (int32_t i = 0; i < groups_buffer_entry_qw_count; ++i) {
    if (i % (key_qw_count + agg_col_count) < key_qw_count) {
      groups_buffer[i] = EMPTY_KEY;
    } else {
      auto init_idx = (i - key_qw_count) % (key_qw_count + agg_col_count);
      const ssize_t bitmap_sz { agg_bitmap_size[init_idx] };
      if (!bitmap_sz) {
        groups_buffer[i] = init_vals[init_idx];
      } else {
        groups_buffer[i] = bitmap_sz > 0 ? allocateCountDistinctBitmap(bitmap_sz) : allocateCountDistinctSet();
      }
    }
  }
}

// deferred is true for group by queries; initGroups will allocate a bitmap
// for each group slot
std::vector<ssize_t> QueryExecutionContext::allocateCountDistinctBuffers(const bool deferred) {
  const size_t agg_col_count { query_mem_desc_.agg_col_widths.size() };
  std::vector<ssize_t> agg_bitmap_size(deferred ? agg_col_count : 0);

  size_t init_agg_idx { 0 };
  for (size_t target_idx = 0; target_idx < executor_->plan_state_->target_exprs_.size(); ++target_idx) {
    const auto target_expr = executor_->plan_state_->target_exprs_[target_idx];
    const auto agg_info = target_info(target_expr);
    if (agg_info.is_distinct) {
      CHECK(agg_info.is_agg && agg_info.agg_kind == kCOUNT);
      auto count_distinct_it = query_mem_desc_.count_distinct_descriptors_.find(target_idx);
      CHECK(count_distinct_it != query_mem_desc_.count_distinct_descriptors_.end());
      const auto& count_distinct_desc = count_distinct_it->second;
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        if (deferred) {
          agg_bitmap_size[init_agg_idx] = count_distinct_desc.bitmap_sz_bits;
        } else {
          init_agg_vals_[init_agg_idx] = allocateCountDistinctBitmap(count_distinct_desc.bitmap_sz_bits);
        }
      } else {
        CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet);
        if (deferred) {
          agg_bitmap_size[init_agg_idx] = -1;
        } else {
          init_agg_vals_[init_agg_idx] = allocateCountDistinctSet();
        }
      }
    }
    if (agg_info.agg_kind == kAVG) {
      init_agg_idx += 2;
    } else {
      ++init_agg_idx;
    }
  }
  CHECK_LE(init_agg_idx, agg_col_count);

  return agg_bitmap_size;
}

int64_t QueryExecutionContext::allocateCountDistinctBitmap(const size_t bitmap_sz) {
  auto bitmap_byte_sz = bitmap_size_bytes(bitmap_sz);
  auto count_distinct_buffer = static_cast<int8_t*>(calloc(bitmap_byte_sz, 1));
  row_set_mem_owner_->addCountDistinctBuffer(count_distinct_buffer);
  return reinterpret_cast<int64_t>(count_distinct_buffer);
}

int64_t QueryExecutionContext::allocateCountDistinctSet() {
  auto count_distinct_set = new std::set<int64_t>();
  row_set_mem_owner_->addCountDistinctSet(count_distinct_set);
  return reinterpret_cast<int64_t>(count_distinct_set);
}

ResultRows QueryExecutionContext::getRowSet(
    const std::vector<Analyzer::Expr*>& targets,
    const bool was_auto_device) const {
  std::vector<ResultRows> results_per_sm;
  CHECK_EQ(num_buffers_, group_by_buffers_.size());
  if (device_type_ == ExecutorDeviceType::CPU) {
    CHECK_EQ(1, num_buffers_);
    return groupBufferToResults(0, targets, was_auto_device);
  }
  size_t step { query_mem_desc_.threadsShareMemory() ? executor_->blockSize() : 1 };
  for (size_t i = 0; i < group_by_buffers_.size(); i += step) {
    results_per_sm.emplace_back(groupBufferToResults(i, targets, was_auto_device));
  }
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return executor_->reduceMultiDeviceResults(results_per_sm, row_set_mem_owner_);
}

namespace {

int64_t lazy_decode(const Analyzer::ColumnVar* col_var, const int8_t* byte_stream, const int64_t pos) {
  const auto enc_type = col_var->get_compression();
  const auto& type_info = col_var->get_type_info();
  if (type_info.is_fp()) {
    double fval = (type_info.get_type() == kFLOAT)
      ? fixed_width_float_decode_noinline(byte_stream, pos)
      : fixed_width_double_decode_noinline(byte_stream, pos);
    return *reinterpret_cast<int64_t*>(&fval);
  }
  CHECK(type_info.is_integer() || type_info.is_time() || type_info.is_boolean() ||
        (type_info.is_string() && enc_type == kENCODING_DICT));
  size_t type_bitwidth = get_bit_width(col_var->get_type_info().get_type());
  if (col_var->get_type_info().get_compression() == kENCODING_FIXED) {
    type_bitwidth = col_var->get_type_info().get_comp_param();
  }
  CHECK_EQ(0, type_bitwidth % 8);
  return fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
}

}  // namespace

ResultRows QueryExecutionContext::groupBufferToResults(
    const size_t i,
    const std::vector<Analyzer::Expr*>& targets,
    const bool was_auto_device) const {
  const size_t group_by_col_count { query_mem_desc_.group_col_widths.size() };
  const size_t agg_col_count { query_mem_desc_.agg_col_widths.size() };
  auto impl = [group_by_col_count, agg_col_count, was_auto_device, this, &targets](
      const size_t groups_buffer_entry_count,
      int64_t* group_by_buffer) {
    if (query_mem_desc_.keyless_hash) {
      CHECK_EQ(1, group_by_col_count);
      CHECK_EQ(targets.size(), agg_col_count);
      const int8_t warp_count = query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1;
      if (!query_mem_desc_.interleavedBins(ExecutorDeviceType::GPU) || !was_auto_device) {
        return ResultRows(targets, executor_, row_set_mem_owner_,
          group_by_buffer, groups_buffer_entry_count, query_mem_desc_.min_val, warp_count);
      }
      // Can't do the fast reduction in auto mode for interleaved bins, warp count isn't the same
      ResultRows results(targets, executor_, row_set_mem_owner_);
      results.addKeylessGroupByBuffer(group_by_buffer, groups_buffer_entry_count, query_mem_desc_.min_val, warp_count);
      return results;
    }
    ResultRows results(targets, executor_, row_set_mem_owner_);
    for (size_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
      const size_t key_off = (group_by_col_count + agg_col_count) * bin;
      if (group_by_buffer[key_off] == EMPTY_KEY) {
        continue;
      }
      size_t out_vec_idx = 0;
      std::vector<int64_t> multi_key;
      for (size_t val_tuple_idx = 0; val_tuple_idx < group_by_col_count; ++val_tuple_idx) {
        const int64_t key_comp = group_by_buffer[key_off + val_tuple_idx];
        CHECK_NE(key_comp, EMPTY_KEY);
        multi_key.push_back(key_comp);
      }
      results.beginRow(multi_key);
      for (const auto target_expr : targets) {
        bool is_real_string = (target_expr && target_expr->get_type_info().is_string() &&
          target_expr->get_type_info().get_compression() == kENCODING_NONE);
        const int global_col_id { dynamic_cast<Analyzer::ColumnVar*>(target_expr)
          ? dynamic_cast<Analyzer::ColumnVar*>(target_expr)->get_column_id()
          : -1 };
        const auto agg_info = target_info(target_expr);
        if (is_real_string) {
          int64_t str_len = group_by_buffer[key_off + out_vec_idx + group_by_col_count + 1];
          int64_t str_ptr = group_by_buffer[key_off + out_vec_idx + group_by_col_count];
          CHECK_GE(str_len, 0);
          if (executor_->plan_state_->isLazyFetchColumn(target_expr)) {  // TODO(alex): expensive!!!, remove
            CHECK_EQ(str_ptr, str_len);  // both are the row index in this case
            VarlenDatum vd;
            bool is_end;
            CHECK_GE(global_col_id, 0);
            auto col_id = executor_->getLocalColumnId(global_col_id, false);
            CHECK_EQ(1, col_buffers_.size());
            auto& frag_col_buffers = col_buffers_.front();
            ChunkIter_get_nth(
              reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffers[col_id])),
              str_ptr, false, &vd, &is_end);
            CHECK(!is_end);
            if (!vd.is_null) {
              results.addValue(std::string(reinterpret_cast<char*>(vd.pointer), vd.length));
            } else {
              results.addValue();
            }
          } else {
            CHECK(device_type_ == ExecutorDeviceType::CPU ||
                  device_type_ == ExecutorDeviceType::GPU);
            if (device_type_ == ExecutorDeviceType::CPU) {
              if (str_ptr) {
                results.addValue(std::string(reinterpret_cast<char*>(str_ptr), str_len));
              } else {
                results.addValue();
              }
            } else {
              if (str_ptr) {
                std::vector<int8_t> cpu_buffer(str_len);
                auto& data_mgr = executor_->catalog_->get_dataMgr();
                copy_from_gpu(&data_mgr, &cpu_buffer[0], static_cast<CUdeviceptr>(str_ptr), str_len, device_id_);
                results.addValue(std::string(reinterpret_cast<char*>(&cpu_buffer[0]), str_len));
              } else {
                results.addValue();
              }
            }
          }
          out_vec_idx += 2;
        } else {
          int64_t val1 = group_by_buffer[key_off + out_vec_idx + group_by_col_count];
          if (executor_->plan_state_->isLazyFetchColumn(target_expr)) {
            CHECK_GE(global_col_id, 0);
            auto col_id = executor_->getLocalColumnId(global_col_id, false);
            CHECK_EQ(1, col_buffers_.size());
            auto& frag_col_buffers = col_buffers_.front();
            val1 = lazy_decode(static_cast<Analyzer::ColumnVar*>(target_expr), frag_col_buffers[col_id], val1);
          }
          if (agg_info.agg_kind == kAVG) {
            CHECK(!executor_->plan_state_->isLazyFetchColumn(target_expr));
            ++out_vec_idx;
            int64_t val2 = group_by_buffer[key_off + out_vec_idx + group_by_col_count];
            results.addValue(val1, val2);
          } else {
            results.addValue(val1);
          }
          ++out_vec_idx;
        }
      }
    }
    return results;
  };
  ResultRows results(targets, executor_, row_set_mem_owner_);
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    results = impl(query_mem_desc_.entry_count_small, small_group_by_buffers_[i]);
  }
  CHECK_LT(i, group_by_buffers_.size());
  auto more_results = impl(query_mem_desc_.entry_count, group_by_buffers_[i]);
  if (query_mem_desc_.keyless_hash) {
    return more_results;
  }
  results.append(more_results);
  return results;
}

std::vector<int64_t*> QueryExecutionContext::launchGpuCode(
    const std::vector<void*>& cu_functions,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<std::vector<const int8_t*>> col_buffers,
    const std::vector<int64_t>& num_rows,
    const int64_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    Data_Namespace::DataMgr* data_mgr,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    int32_t* error_code) const {
  data_mgr->cudaMgr_->setContext(device_id);
  auto cu_func = static_cast<CUfunction>(cu_functions[device_id]);
  std::vector<int64_t*> out_vec;
  uint32_t num_fragments = col_buffers.size();
  CUdeviceptr multifrag_col_buffers_dev_ptr;
  const size_t col_count { num_fragments > 0 ? col_buffers.front().size() : 0 };
  if (col_count) {
    std::vector<CUdeviceptr> multifrag_col_dev_buffers;
    for (auto frag_col_buffers : col_buffers) {
      std::vector<CUdeviceptr> col_dev_buffers;
      for (auto col_buffer : frag_col_buffers) {
        col_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(col_buffer));
      }
      auto col_buffers_dev_ptr = alloc_gpu_mem(
        data_mgr, col_count * sizeof(CUdeviceptr), device_id);
      copy_to_gpu(data_mgr, col_buffers_dev_ptr, &col_dev_buffers[0],
        col_count * sizeof(CUdeviceptr), device_id);
      multifrag_col_dev_buffers.push_back(col_buffers_dev_ptr);
    }
    multifrag_col_buffers_dev_ptr = alloc_gpu_mem(data_mgr, num_fragments * sizeof(CUdeviceptr), device_id);
    copy_to_gpu(data_mgr, multifrag_col_buffers_dev_ptr, &multifrag_col_dev_buffers[0],
      num_fragments * sizeof(CUdeviceptr), device_id);
  }
  CUdeviceptr literals_dev_ptr { 0 };
  if (!literal_buff.empty()) {
    CHECK(hoist_literals);
    literals_dev_ptr = alloc_gpu_mem(data_mgr, literal_buff.size(), device_id);
    copy_to_gpu(data_mgr, literals_dev_ptr, &literal_buff[0], literal_buff.size(), device_id);
  }
  CUdeviceptr num_rows_dev_ptr { 0 };
  {
    num_rows_dev_ptr = alloc_gpu_mem(data_mgr, sizeof(int64_t) * num_rows.size(), device_id);
    copy_to_gpu(data_mgr, num_rows_dev_ptr, &num_rows[0],
      sizeof(int64_t) * num_rows.size(), device_id);
  }
  CUdeviceptr num_fragments_dev_ptr { 0 };
  {
    num_fragments_dev_ptr = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id);
    copy_to_gpu(data_mgr, num_fragments_dev_ptr, &num_fragments, sizeof(uint32_t), device_id);
  }
  CUdeviceptr max_matched_dev_ptr { 0 };
  {
    int64_t max_matched { scan_limit };
    max_matched_dev_ptr = alloc_gpu_mem(data_mgr, sizeof(int64_t), device_id);
    copy_to_gpu(data_mgr, max_matched_dev_ptr, &max_matched,
      sizeof(int64_t), device_id);
  }
  CUdeviceptr init_agg_vals_dev_ptr;
  {
    init_agg_vals_dev_ptr = alloc_gpu_mem(
      data_mgr, init_agg_vals.size() * sizeof(int64_t), device_id);
    copy_to_gpu(data_mgr, init_agg_vals_dev_ptr, &init_agg_vals[0],
      init_agg_vals.size() * sizeof(int64_t), device_id);
  }
  std::vector<int32_t> error_codes(block_size_x);
  auto error_code_dev_ptr = alloc_gpu_mem(data_mgr, grid_size_x * sizeof(error_codes[0]), device_id);
  copy_to_gpu(data_mgr, error_code_dev_ptr, &error_codes[0], grid_size_x * sizeof(error_codes[0]), device_id);
  {
    const unsigned block_size_y = 1;
    const unsigned block_size_z = 1;
    const unsigned grid_size_y  = 1;
    const unsigned grid_size_z  = 1;
    if (query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU) > 0) {  // group by path
      CHECK(!group_by_buffers_.empty());
      auto gpu_query_mem = create_dev_group_by_buffers(
        data_mgr, group_by_buffers_, small_group_by_buffers_, query_mem_desc_,
        block_size_x, grid_size_x, device_id);
      if (hoist_literals) {
        void* kernel_params[] = {
          &multifrag_col_buffers_dev_ptr,
          &num_fragments_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &max_matched_dev_ptr,
          &init_agg_vals_dev_ptr,
          &gpu_query_mem.group_by_buffers.first,
          &gpu_query_mem.small_group_by_buffers.first,
          &error_code_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                       nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &multifrag_col_buffers_dev_ptr,
          &num_fragments_dev_ptr,
          &num_rows_dev_ptr,
          &max_matched_dev_ptr,
          &init_agg_vals_dev_ptr,
          &gpu_query_mem.group_by_buffers.first,
          &gpu_query_mem.small_group_by_buffers.first,
          &error_code_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                       nullptr, kernel_params, nullptr));
      }
      copy_group_by_buffers_from_gpu(data_mgr, this, gpu_query_mem, block_size_x, grid_size_x, device_id);
      copy_from_gpu(data_mgr, &error_codes[0], error_code_dev_ptr, grid_size_x * sizeof(error_codes[0]), device_id);
      *error_code = 0;
      for (const auto err : error_codes) {
        if (err && (!*error_code || err > *error_code)) {
          *error_code = err;
          break;
        }
      }
    } else {
      std::vector<CUdeviceptr> out_vec_dev_buffers;
      const size_t agg_col_count { init_agg_vals.size() };
      for (size_t i = 0; i < agg_col_count; ++i) {
        auto out_vec_dev_buffer = alloc_gpu_mem(
          data_mgr, block_size_x * grid_size_x * sizeof(int64_t) * num_fragments, device_id);
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
      }
      auto out_vec_dev_ptr = alloc_gpu_mem(data_mgr, agg_col_count * sizeof(CUdeviceptr), device_id);
      copy_to_gpu(data_mgr, out_vec_dev_ptr, &out_vec_dev_buffers[0],
        agg_col_count * sizeof(CUdeviceptr), device_id);
      CUdeviceptr unused_dev_ptr { 0 };
      if (hoist_literals) {
        void* kernel_params[] = {
          &multifrag_col_buffers_dev_ptr,
          &num_fragments_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &max_matched_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr,
          &unused_dev_ptr,
          &error_code_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &multifrag_col_buffers_dev_ptr,
          &num_fragments_dev_ptr,
          &num_rows_dev_ptr,
          &max_matched_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr,
          &unused_dev_ptr,
          &error_code_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      }
      for (size_t i = 0; i < agg_col_count; ++i) {
        int64_t* host_out_vec = new int64_t[block_size_x * grid_size_x * sizeof(int64_t) * num_fragments];
        copy_from_gpu(data_mgr, host_out_vec, out_vec_dev_buffers[i],
          block_size_x * grid_size_x * sizeof(int64_t) * num_fragments,
          device_id);
        out_vec.push_back(host_out_vec);
      }
    }
  }
  return out_vec;
}

std::unique_ptr<QueryExecutionContext> QueryMemoryDescriptor::getQueryExecutionContext(
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const int device_id,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const {
  return std::unique_ptr<QueryExecutionContext>(
    new QueryExecutionContext(*this, init_agg_vals, executor, device_type, device_id, col_buffers, row_set_mem_owner));
}

size_t QueryMemoryDescriptor::getBufferSizeQuad(const ExecutorDeviceType device_type) const {
  if (keyless_hash) {
    CHECK_EQ(1, group_col_widths.size());
    return (interleavedBins(device_type)
      ? executor_->warpSize() * agg_col_widths.size()
      : agg_col_widths.size()) * entry_count;
  }
  return (group_col_widths.size() + agg_col_widths.size()) * entry_count;
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

int32_t get_agg_count(const Planner::Plan* plan) {
  int32_t agg_count { 0 };
  const auto target_exprs = get_agg_target_exprs(plan);
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

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getColRangeInfo(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan_);
  const int64_t guessed_range_max { 255 };  // TODO(alex): replace with educated guess
  if (!agg_plan) {
    return { GroupByColRangeType::Scan, 0, guessed_range_max, false };
  }
  const auto& groupby_exprs = agg_plan->get_groupby_list();
  if (groupby_exprs.size() != 1) {
    try {
      checked_int64_t cardinality { 1 };
      bool has_nulls { false };
      for (const auto groupby_expr : groupby_exprs) {
        auto col_range_info = getExprRangeInfo(groupby_expr, fragments);
        if (col_range_info.hash_type_ != GroupByColRangeType::OneColKnownRange) {
          return { GroupByColRangeType::MultiCol, 0, 0, false };
        }
        auto crt_col_cardinality = col_range_info.max - col_range_info.min + 1 + (col_range_info.has_nulls ? 1 : 0);
        CHECK_GT(crt_col_cardinality, 0);
        cardinality *= crt_col_cardinality;
        if (col_range_info.has_nulls) {
          has_nulls = true;
        }
      }
      if (cardinality > 10000000) {  // more than 10M groups is a lot
        return { GroupByColRangeType::MultiCol, 0, 0, false };
      }
      return { GroupByColRangeType::MultiColPerfectHash, 0, int64_t(cardinality), has_nulls };
    } catch (...) {  // overflow when computing cardinality
      return { GroupByColRangeType::MultiCol, 0, 0, false };
    }
  }
  return getExprRangeInfo(groupby_exprs.front(), fragments);
}

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getExprRangeInfo(
    const Analyzer::Expr* expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const int64_t guessed_range_max { 255 };  // TODO(alex): replace with educated guess

  const auto expr_range = getExpressionRange(expr, fragments, executor_);
  switch (expr_range.type) {
  case ExpressionRangeType::Integer:
    return { GroupByColRangeType::OneColKnownRange, expr_range.int_min, expr_range.int_max, expr_range.has_nulls };
  case ExpressionRangeType::Invalid:
  case ExpressionRangeType::FloatingPoint:
    return { GroupByColRangeType::OneColGuessedRange, 0, guessed_range_max, false };
  default:
    CHECK(false);
  }
  CHECK(false);
  return { GroupByColRangeType::Scan, 0, 0, false };
}

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->ll_int(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

namespace {

std::list<Analyzer::Expr*> group_by_exprs(const Planner::Plan* plan) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
  // For non-aggregate (scan only) plans, execute them like a group by
  // row index -- the null pointer means row index to Executor::codegen().
  return agg_plan
    ? agg_plan->get_groupby_list()
    : std::list<Analyzer::Expr*> { nullptr };
}

}  // namespace

GroupByAndAggregate::GroupByAndAggregate(
    Executor* executor,
    const Planner::Plan* plan,
    const Fragmenter_Namespace::QueryInfo& query_info,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const size_t max_groups_buffer_entry_count,
    const int64_t scan_limit)
  : executor_(executor)
  , plan_(plan)
  , query_info_(query_info)
  , row_set_mem_owner_(row_set_mem_owner)
  , max_groups_buffer_entry_count_(max_groups_buffer_entry_count)
  , scan_limit_(scan_limit) {
  CHECK(plan_);
  for (const auto groupby_expr : group_by_exprs(plan_)) {
    if (!groupby_expr) {
      continue;
    }
    const auto& groupby_ti = groupby_expr->get_type_info();
    if (groupby_ti.is_string() && groupby_ti.get_compression() != kENCODING_DICT) {
      throw std::runtime_error("Group by not supported for none-encoding strings");
    }
  }
}

QueryMemoryDescriptor GroupByAndAggregate::getQueryMemoryDescriptor(const size_t max_groups_buffer_entry_count) {
  auto group_cols = group_by_exprs(plan_);
  for (const auto group_expr : group_cols) {
    const auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(group_expr);
    if (!case_expr) {
      continue;
    }
    Analyzer::DomainSet domain_set;
    case_expr->get_domain(domain_set);
    if (domain_set.empty()) {
      continue;
    }
    const auto& case_ti = case_expr->get_type_info();
    if (case_ti.is_string()) {
      CHECK_EQ(kENCODING_DICT, case_ti.get_compression());
      auto sd = executor_->getStringDictionary(case_ti.get_comp_param(), row_set_mem_owner_);
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
  auto group_col_widths = get_col_byte_widths(group_cols);
  const auto& target_list = plan_->get_targetlist();
  std::vector<Analyzer::Expr*> target_expr_list;
  CountDistinctDescriptors count_distinct_descriptors;
  size_t target_idx { 0 };
  for (const auto target : target_list) {
    auto target_expr = target->get_expr();
    auto agg_info = target_info(target_expr);
    if (agg_info.is_distinct) {
      CHECK(agg_info.is_agg);
      CHECK_EQ(kCOUNT, agg_info.agg_kind);
      const auto agg_expr = static_cast<const Analyzer::AggExpr*>(target_expr);
      const auto& arg_ti = agg_expr->get_arg()->get_type_info();
      if (arg_ti.is_string() && arg_ti.get_compression() != kENCODING_DICT) {
        throw std::runtime_error("Count distinct not supported for none-encoding strings");
      }
      auto arg_range_info = getExprRangeInfo(agg_expr->get_arg(), query_info_.fragments);
      CountDistinctImplType count_distinct_impl_type { CountDistinctImplType::StdSet };
      int64_t bitmap_sz_bits { 0 };
      if (arg_range_info.hash_type_ == GroupByColRangeType::OneColKnownRange) {
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
        bitmap_sz_bits = arg_range_info.max - arg_range_info.min + 1;
        if (bitmap_sz_bits <= 0) {
          count_distinct_impl_type = CountDistinctImplType::StdSet;
        }
      }
      CountDistinctDescriptor count_distinct_desc {
        executor_,
        count_distinct_impl_type,
        arg_range_info.min,
        bitmap_sz_bits
      };
      auto it_ok = count_distinct_descriptors.insert(std::make_pair(target_idx, count_distinct_desc));
      CHECK(it_ok.second);
    }
    target_expr_list.push_back(target_expr);
    ++target_idx;
  }
  if (!count_distinct_descriptors.empty()) {
    CHECK(row_set_mem_owner_);
    row_set_mem_owner_->setCountDistinctDescriptors(count_distinct_descriptors);
  }
  auto agg_col_widths = get_col_byte_widths(target_expr_list);

  if (group_col_widths.empty()) {
    return {
      executor_,
      GroupByColRangeType::Scan, false, false,
      group_col_widths, agg_col_widths,
      0, 0,
      0, 0, false,
      GroupByMemSharing::Private,
      count_distinct_descriptors
    };
  }

  const auto col_range_info = getColRangeInfo(query_info_.fragments);

  switch (col_range_info.hash_type_) {
  case GroupByColRangeType::OneColKnownRange:
  case GroupByColRangeType::OneColGuessedRange:
  case GroupByColRangeType::Scan: {
    if (col_range_info.hash_type_ == GroupByColRangeType::OneColGuessedRange ||
        col_range_info.hash_type_ == GroupByColRangeType::Scan ||
        col_range_info.max >= col_range_info.min + static_cast<int64_t>(max_groups_buffer_entry_count)) {
      return {
        executor_,
        col_range_info.hash_type_, false, false,
        group_col_widths, agg_col_widths,
        max_groups_buffer_entry_count,
        scan_limit_ ? scan_limit_ : executor_->small_groups_buffer_entry_count_,
        col_range_info.min, col_range_info.max, col_range_info.has_nulls,
        GroupByMemSharing::Shared,
        count_distinct_descriptors
      };
    } else {
      bool keyless = true;
      bool found_count = false;  // shouldn't use keyless for projection only
      if (keyless) {
        for (const auto target_expr : target_expr_list) {
          auto agg_info = target_info(target_expr);
          if (agg_info.is_agg) {
            if (agg_info.agg_kind != kCOUNT) {
              keyless = false;
              break;
            } else {
              found_count = true;
            }
          }
          if (!agg_info.is_agg && !dynamic_cast<Analyzer::ColumnVar*>(target_expr)) {
            keyless = false;
            break;
          }
        }
      }
      if (!found_count) {
        keyless = false;
      }
      const size_t bin_count = col_range_info.max - col_range_info.min + 1 + (col_range_info.has_nulls ? 1 : 0);
      const size_t interleaved_max_threshold { 20 };
      bool interleaved_bins = keyless &&
        bin_count <= interleaved_max_threshold;
      return {
        executor_,
        col_range_info.hash_type_, keyless, interleaved_bins,
        group_col_widths, agg_col_widths,
        bin_count, 0,
        col_range_info.min, col_range_info.max, col_range_info.has_nulls,
        GroupByMemSharing::Shared,
        count_distinct_descriptors
      };
    }
  }
  case GroupByColRangeType::MultiCol: {
    return {
      executor_,
      col_range_info.hash_type_, false, false,
      group_col_widths, agg_col_widths,
      max_groups_buffer_entry_count, 0,
      0, 0, false,
      GroupByMemSharing::Shared,
      count_distinct_descriptors
    };
  }
  case GroupByColRangeType::MultiColPerfectHash: {
    return {
      executor_,
      col_range_info.hash_type_, false, false,
      group_col_widths, agg_col_widths,
      static_cast<size_t>(col_range_info.max), 0,
      0, 0, col_range_info.has_nulls,
      GroupByMemSharing::Shared,
      count_distinct_descriptors
    };
  }
  default:
    CHECK(false);
  }
  CHECK(false);
  return {};
}

bool QueryMemoryDescriptor::usesGetGroupValueFast() const {
  return (hash_type == GroupByColRangeType::OneColKnownRange && !getSmallBufferSizeBytes());
}

bool QueryMemoryDescriptor::usesCachedContext() const {
  return usesGetGroupValueFast() || hash_type == GroupByColRangeType::MultiColPerfectHash;
}

bool QueryMemoryDescriptor::threadsShareMemory() const {
  return sharing == GroupByMemSharing::Shared;
}

bool QueryMemoryDescriptor::lazyInitGroups(const ExecutorDeviceType device_type) const {
  return device_type == ExecutorDeviceType::GPU && hash_type == GroupByColRangeType::MultiCol;
}

bool QueryMemoryDescriptor::interleavedBins(const ExecutorDeviceType device_type) const {
  return interleaved_bins_on_gpu && device_type == ExecutorDeviceType::GPU;
}

size_t QueryMemoryDescriptor::sharedMemBytes(const ExecutorDeviceType device_type) const {
  CHECK(device_type == ExecutorDeviceType::CPU || device_type == ExecutorDeviceType::GPU);
  if (device_type == ExecutorDeviceType::CPU) {
    return 0;
  }
  const size_t shared_mem_threshold { 0 };
  const size_t shared_mem_bytes { getBufferSizeBytes(ExecutorDeviceType::GPU) };
  if (!usesGetGroupValueFast() || shared_mem_bytes > shared_mem_threshold) {
    return 0;
  }
  return shared_mem_bytes;
}

GroupByAndAggregate::DiamondCodegen::DiamondCodegen(
    llvm::Value* cond,
    Executor* executor,
    const bool chain_to_next,
    DiamondCodegen* parent)
  : executor_(executor)
  , chain_to_next_(chain_to_next)
  , parent_(parent) {
  if (parent_) {
    CHECK(!chain_to_next_);
  }
  cond_true_ = llvm::BasicBlock::Create(
    LL_CONTEXT, "cond_true", ROW_FUNC);
  cond_false_ = llvm::BasicBlock::Create(
    LL_CONTEXT, "cond_false", ROW_FUNC);

  LL_BUILDER.CreateCondBr(cond, cond_true_, cond_false_);
  LL_BUILDER.SetInsertPoint(cond_true_);
}

void GroupByAndAggregate::DiamondCodegen::setChainToNext() {
  CHECK(!parent_);
  chain_to_next_ = true;
}

GroupByAndAggregate::DiamondCodegen::~DiamondCodegen() {
  if (parent_) {
    LL_BUILDER.CreateBr(parent_->cond_false_);
  } else if (chain_to_next_) {
    LL_BUILDER.CreateBr(cond_false_);
  }
  LL_BUILDER.SetInsertPoint(cond_false_);
}

bool GroupByAndAggregate::codegen(
    llvm::Value* filter_result,
    const ExecutorDeviceType device_type,
    const bool hoist_literals) {
  CHECK(filter_result);

  bool can_return_error = false;

  {
    const bool is_group_by = !group_by_exprs(plan_).empty();
    auto query_mem_desc = getQueryMemoryDescriptor(max_groups_buffer_entry_count_);

    DiamondCodegen filter_cfg(filter_result, executor_, !is_group_by || query_mem_desc.usesGetGroupValueFast());

    if (is_group_by) {
      if (scan_limit_) {
        auto arg_it = ROW_FUNC->arg_begin();
        ++arg_it;
        ++arg_it;
        auto crt_matched = LL_BUILDER.CreateLoad(arg_it);
        LL_BUILDER.CreateStore(LL_BUILDER.CreateAdd(crt_matched, executor_->ll_int(int64_t(1))), arg_it);
      }

      auto agg_out_start_ptr = codegenGroupBy(query_mem_desc, device_type, hoist_literals);
      if (query_mem_desc.usesGetGroupValueFast() || query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
        if (query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
          filter_cfg.setChainToNext();
        }
        // Don't generate null checks if the group slot is guaranteed to be non-null,
        // as it's the case for get_group_value_fast* family.
        codegenAggCalls(agg_out_start_ptr, {}, query_mem_desc, device_type, hoist_literals);
      } else {
        {
          DiamondCodegen nullcheck_cfg(
            LL_BUILDER.CreateICmpNE(
              agg_out_start_ptr,
              llvm::ConstantPointerNull::get(llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0))),
            executor_,
            false,
            &filter_cfg);
          codegenAggCalls(agg_out_start_ptr, {}, query_mem_desc, device_type, hoist_literals);
        }
        can_return_error = true;
        LL_BUILDER.CreateRet(LL_BUILDER.CreateNeg(LL_BUILDER.CreateTrunc(
          // TODO(alex): remove the trunc once pos is converted to 32 bits
          executor_->cgen_state_->getCurrentRowIndex(), get_int_type(32, LL_CONTEXT))));
      }
    } else {
      auto arg_it = ROW_FUNC->arg_begin();
      std::vector<llvm::Value*> agg_out_vec;
      for (int32_t i = 0; i < get_agg_count(plan_); ++i) {
        agg_out_vec.push_back(arg_it++);
      }
      codegenAggCalls(nullptr, agg_out_vec, query_mem_desc, device_type, hoist_literals);
    }
  }

  LL_BUILDER.CreateRet(LL_INT(0));

  return can_return_error;
}

llvm::Value* GroupByAndAggregate::codegenGroupBy(
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    const bool hoist_literals) {
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;

  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan_);

  // For non-aggregate (scan only) plans, execute them like a group by
  // row index -- the null pointer means row index to Executor::codegen().
  const auto groupby_list = agg_plan
    ? agg_plan->get_groupby_list()
    : std::list<Analyzer::Expr*> { nullptr };

  switch (query_mem_desc.hash_type) {
  case GroupByColRangeType::OneColKnownRange:
  case GroupByColRangeType::OneColGuessedRange:
  case GroupByColRangeType::Scan: {
    CHECK_EQ(1, groupby_list.size());
    const auto group_expr = groupby_list.front();
    const auto group_expr_lv = executor_->groupByColumnCodegen(group_expr, hoist_literals,
      query_mem_desc.has_nulls, query_mem_desc.max_val + 1);
    auto small_groups_buffer = arg_it;
    if (query_mem_desc.usesGetGroupValueFast()) {
      std::string get_group_fn_name { "get_group_value_fast" };
      if (query_mem_desc.keyless_hash) {
        get_group_fn_name += "_keyless";
      }
      if (query_mem_desc.interleavedBins(device_type)) {
        CHECK(query_mem_desc.keyless_hash);
        get_group_fn_name += "_semiprivate";
      }
      std::vector<llvm::Value*> get_group_fn_args {
        groups_buffer,
        group_expr_lv,
        LL_INT(query_mem_desc.min_val)
      };
      if (!query_mem_desc.keyless_hash) {
        get_group_fn_args.push_back(LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size())));
      } else {
        get_group_fn_args.push_back(LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size())));
        if (query_mem_desc.interleavedBins(device_type)) {
          auto warp_idx = emitCall("thread_warp_idx", { LL_INT(executor_->warpSize()) });
          get_group_fn_args.push_back(warp_idx);
          get_group_fn_args.push_back(LL_INT(executor_->warpSize()));
        }
      }
      return emitCall(get_group_fn_name, get_group_fn_args);
    } else {
      ++arg_it;
      return emitCall(
        "get_group_value_one_key",
        {
          groups_buffer,
          LL_INT(static_cast<int32_t>(query_mem_desc.entry_count)),
          small_groups_buffer,
          LL_INT(static_cast<int32_t>(query_mem_desc.entry_count_small)),
          group_expr_lv,
          LL_INT(query_mem_desc.min_val),
          LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size())),
          ++arg_it
        });
    }
    break;
  }
  case GroupByColRangeType::MultiCol:
  case GroupByColRangeType::MultiColPerfectHash: {
    auto key_size_lv = LL_INT(static_cast<int32_t>(query_mem_desc.group_col_widths.size()));
    // create the key buffer
    auto group_key = LL_BUILDER.CreateAlloca(
      llvm::Type::getInt64Ty(LL_CONTEXT),
      key_size_lv);
    int32_t subkey_idx = 0;
    for (const auto group_expr : groupby_list) {
      auto col_range_info = getExprRangeInfo(group_expr, query_info_.fragments);
      const auto group_expr_lv = executor_->groupByColumnCodegen(group_expr, hoist_literals,
        col_range_info.has_nulls, col_range_info.max + 1);
      // store the sub-key to the buffer
      LL_BUILDER.CreateStore(group_expr_lv, LL_BUILDER.CreateGEP(group_key, LL_INT(subkey_idx++)));
    }
    ++arg_it;
    auto perfect_hash_func = query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash
      ? codegenPerfectHashFunction()
      : nullptr;
    if (perfect_hash_func) {
      auto hash_lv = LL_BUILDER.CreateCall(perfect_hash_func, std::vector<llvm::Value*> { group_key, key_size_lv });
      return emitCall("get_matching_group_value_perfect_hash", {
        groups_buffer,
        hash_lv,
        group_key,
        key_size_lv,
        LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size()))
      });
    }
    return emitCall(
      "get_group_value",
      {
        groups_buffer,
        LL_INT(static_cast<int32_t>(query_mem_desc.entry_count)),
        group_key,
        key_size_lv,
        LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size())),
        ++arg_it
      });
    break;
  }
  default:
    CHECK(false);
    break;
  }

  CHECK(false);
  return nullptr;
}

llvm::Function* GroupByAndAggregate::codegenPerfectHashFunction() {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan_);
  CHECK(agg_plan);
  const auto& groupby_exprs = agg_plan->get_groupby_list();
  CHECK_GT(groupby_exprs.size(), 1);
  auto ft = llvm::FunctionType::get(
    get_int_type(32, LL_CONTEXT), std::vector<llvm::Type*> {
      llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0),
      get_int_type(32, LL_CONTEXT)
    },
    false);
  auto key_hash_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
    "perfect_key_hash",
    executor_->cgen_state_->module_);
  key_hash_func->addAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline);
  auto& key_buff_arg = key_hash_func->getArgumentList().front();
  llvm::Value* key_buff_lv = &key_buff_arg;
  auto bb = llvm::BasicBlock::Create(LL_CONTEXT, "entry", key_hash_func);
  llvm::IRBuilder<> key_hash_func_builder(bb);
  llvm::Value* hash_lv { llvm::ConstantInt::get(get_int_type(64, LL_CONTEXT), 0) };
  std::vector<int64_t> cardinalities;
  for (const auto groupby_expr : groupby_exprs) {
    auto col_range_info = getExprRangeInfo(groupby_expr, query_info_.fragments);
    CHECK(col_range_info.hash_type_ == GroupByColRangeType::OneColKnownRange);
    cardinalities.push_back(col_range_info.max - col_range_info.min + 1);
  }
  size_t dim_idx = 0;
  for (const auto groupby_expr : groupby_exprs) {
    auto key_comp_lv = key_hash_func_builder.CreateLoad(
      key_hash_func_builder.CreateGEP(key_buff_lv, LL_INT(dim_idx)));
    auto col_range_info = getExprRangeInfo(groupby_expr, query_info_.fragments);
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
    if (target_info.sql_type.is_string() && target_info.sql_type.get_compression() == kENCODING_NONE) {
      return { "agg_id", "agg_id" };
    }
    return { "agg_id" };
  }
  switch (target_info.agg_kind) {
  case kAVG:
    return { "agg_sum", "agg_count" };
  case kCOUNT:
    return { target_info.is_distinct ? "agg_count_distinct" : "agg_count" };
  case kMAX:
    return { "agg_max" };
  case kMIN:
    return { "agg_min" };
  case kSUM:
    return { "agg_sum" };
  default:
    CHECK(false);
  }
}

}  // namespace

void GroupByAndAggregate::codegenAggCalls(
    llvm::Value* agg_out_start_ptr,
    const std::vector<llvm::Value*>& agg_out_vec,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    const bool hoist_literals) {
  // TODO(alex): unify the two cases, the output for non-group by queries
  //             should be a contiguous buffer
  const bool is_group_by { agg_out_start_ptr };
  if (is_group_by) {
    CHECK(agg_out_vec.empty());
  } else {
    CHECK(!agg_out_vec.empty());
  }

  const auto& target_list = plan_->get_targetlist();
  int32_t agg_out_off { 0 };
  for (size_t target_idx = 0; target_idx < target_list.size(); ++target_idx) {
    auto target = target_list[target_idx];
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    const auto agg_info = target_info(target_expr);
    const auto agg_fn_names = agg_fn_base_names(agg_info);
    auto target_lvs = codegenAggArg(target_expr, hoist_literals);
    if (target_lvs.size() == 3) {
      // none encoding string, pop the packed pointer + length since
      // it's only useful for IS NULL checks and assumed to be only
      // two components (pointer and length) for the purpose of projection
      target_lvs.erase(target_lvs.begin());
    }
    if (target_lvs.size() < agg_fn_names.size()) {
      CHECK_EQ(1, target_lvs.size());
      CHECK_EQ(2, agg_fn_names.size());
      for (size_t i = 1; i < agg_fn_names.size(); ++i) {
        target_lvs.push_back(target_lvs.front());
      }
    } else {
      CHECK_EQ(agg_fn_names.size(), target_lvs.size());
      CHECK(target_lvs.size() == 1 || target_lvs.size() == 2);
    }
    const bool is_simple_count = agg_info.is_agg && agg_info.agg_kind == kCOUNT && !agg_info.is_distinct;
    if (device_type == ExecutorDeviceType::GPU && query_mem_desc.threadsShareMemory() && is_simple_count) {
      CHECK_EQ(1, agg_fn_names.size());
      // TODO(alex): use 32-bit wherever possible, avoid casts
      auto acc_i32 = LL_BUILDER.CreateCast(
        llvm::Instruction::CastOps::BitCast,
        is_group_by
          ? LL_BUILDER.CreateGEP(agg_out_start_ptr, LL_INT(agg_out_off))
          : agg_out_vec[agg_out_off],
        llvm::PointerType::get(get_int_type(32, LL_CONTEXT), 0));
      LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add, acc_i32, LL_INT(1),
        llvm::AtomicOrdering::Monotonic);
      ++agg_out_off;
      continue;
    }
    size_t target_lv_idx = 0;
    for (const auto& agg_base_name : agg_fn_names) {
      auto target_lv = executor_->toDoublePrecision(target_lvs[target_lv_idx]);
      std::vector<llvm::Value*> agg_args {
        is_group_by
          ? LL_BUILDER.CreateGEP(agg_out_start_ptr, LL_INT(agg_out_off))
          : agg_out_vec[agg_out_off],
        // TODO(alex): simply use target_lv once we're done with refactoring,
        //             for now just generate the same IR for easy debugging
        is_simple_count ? LL_INT(0L) : target_lv
      };
      std::string agg_fname { agg_base_name };
      if (agg_info.sql_type.is_fp()) {
        if (!executor_->plan_state_->isLazyFetchColumn(target_expr)) {
          CHECK(target_lv->getType()->isDoubleTy());
          agg_fname += "_double";
        }
      }
      if (agg_info.is_distinct) {
        CHECK(!agg_info.sql_type.is_fp());
        CHECK_EQ("agg_count_distinct", agg_base_name);
        codegenCountDistinct(target_idx, target_expr, agg_args, query_mem_desc, device_type, is_group_by, agg_out_off);
      } else {
        if (agg_info.skip_null_val) {
          agg_fname += "_skip_val";
          auto null_lv = executor_->toDoublePrecision(agg_info.sql_type.is_fp()
            ? static_cast<llvm::Value*>(executor_->inlineFpNull(agg_info.sql_type))
            : static_cast<llvm::Value*>(executor_->inlineIntNull(agg_info.sql_type)));
          agg_args.push_back(null_lv);
        }
        if (!agg_info.is_distinct) {
          emitCall(
            (device_type == ExecutorDeviceType::GPU && query_mem_desc.threadsShareMemory())
              ? agg_fname + "_shared"
              : agg_fname,
            agg_args);
        }
      }
      ++agg_out_off;
      ++target_lv_idx;
    }
  }
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    executor_->plan_state_->isLazyFetchColumn(target_expr);
  }
}

void GroupByAndAggregate::codegenCountDistinct(
    const size_t target_idx,
    const Analyzer::Expr* target_expr,
    std::vector<llvm::Value*>& agg_args,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    const bool is_group_by,
    const int32_t agg_out_off) {
  const auto agg_info = target_info(target_expr);
  const auto& arg_ti = static_cast<const Analyzer::AggExpr*>(target_expr)->get_arg()->get_type_info();
  if (arg_ti.is_fp()) {
    agg_args.back() = executor_->cgen_state_->ir_builder_.CreateBitCast(agg_args.back(),
      get_int_type(64, executor_->cgen_state_->context_));
  }
  CHECK(device_type == ExecutorDeviceType::CPU);
  auto it_count_distinct = query_mem_desc.count_distinct_descriptors_.find(target_idx);
  CHECK(it_count_distinct != query_mem_desc.count_distinct_descriptors_.end());
  std::string agg_fname { "agg_count_distinct" };
  if (it_count_distinct->second.impl_type_ == CountDistinctImplType::Bitmap) {
    agg_fname += "_bitmap";
    agg_args.push_back(LL_INT(static_cast<int64_t>(it_count_distinct->second.min_val)));
  }
  if (agg_info.skip_null_val) {
    auto null_lv = executor_->toDoublePrecision(arg_ti.is_fp()
      ? static_cast<llvm::Value*>(executor_->inlineFpNull(arg_ti))
      : static_cast<llvm::Value*>(executor_->inlineIntNull(arg_ti)));
    null_lv = executor_->cgen_state_->ir_builder_.CreateBitCast(
      null_lv, get_int_type(64, executor_->cgen_state_->context_));
    agg_fname += "_skip_val";
    agg_args.push_back(null_lv);
  }
  if (it_count_distinct->second.impl_type_ == CountDistinctImplType::Bitmap) {
    emitCall(agg_fname, agg_args);
  } else {
    emitCall(agg_fname, agg_args);
  }
}

std::vector<llvm::Value*> GroupByAndAggregate::codegenAggArg(
    const Analyzer::Expr* target_expr,
    const bool hoist_literals) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  return agg_expr
    ? executor_->codegen(agg_expr->get_arg(), true, hoist_literals)
    : executor_->codegen(target_expr, !executor_->plan_state_->allow_lazy_fetch_, hoist_literals);
}

llvm::Value* GroupByAndAggregate::emitCall(
    const std::string& fname,
    const std::vector<llvm::Value*>& args) {
  return executor_->cgen_state_->emitCall(fname, args);
}

#undef ROW_FUNC
#undef LL_INT
#undef LL_BUILDER
#undef LL_CONTEXT
