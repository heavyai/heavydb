/*
 * Copyright 2018 OmniSci, Inc.
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

#include "WindowContext.h"
#include <numeric>
#include "../Shared/checked_alloc.h"
#include "../Shared/sql_window_function_to_string.h"
#include "Descriptors/CountDistinctDescriptor.h"
#include "OutputBufferInitialization.h"
#include "ResultSetBufferAccessors.h"
#include "RuntimeFunctions.h"
#include "TypePunning.h"

WindowFunctionContext::WindowFunctionContext(
    const Analyzer::WindowFunction* window_func,
    const std::shared_ptr<JoinHashTableInterface>& partitions,
    const size_t elem_count,
    const ExecutorDeviceType device_type)
    : window_func_(window_func)
    , partitions_(partitions)
    , elem_count_(elem_count)
    , output_(nullptr)
    , partition_start_(nullptr)
    , partition_end_(nullptr)
    , device_type_(device_type) {}

WindowFunctionContext::~WindowFunctionContext() {
  free(output_);
  free(partition_start_);
  free(partition_end_);
}

void WindowFunctionContext::addOrderColumn(
    const int8_t* column,
    const Analyzer::ColumnVar* col_var,
    const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner) {
  order_columns_owner_.push_back(chunks_owner);
  order_columns_.push_back(column);
}

namespace {

// Converts the sorted indices to a mapping from row position to row number.
std::vector<int64_t> index_to_row_number(const int64_t* index, const size_t index_size) {
  std::vector<int64_t> row_numbers(index_size);
  for (size_t i = 0; i < index_size; ++i) {
    row_numbers[index[i]] = i + 1;
  }
  return row_numbers;
}

// Returns true iff the current element is greater than the previous, according to the
// comparator. This is needed because peer rows have to have the same rank.
bool advance_current_rank(
    const std::function<bool(const int64_t lhs, const int64_t rhs)>& comparator,
    const int64_t* index,
    const size_t i) {
  if (i == 0) {
    return false;
  }
  return comparator(index[i - 1], index[i]);
}

// Computes the mapping from row position to rank.
std::vector<int64_t> index_to_rank(
    const int64_t* index,
    const size_t index_size,
    const std::function<bool(const int64_t lhs, const int64_t rhs)>& comparator) {
  std::vector<int64_t> rank(index_size);
  size_t crt_rank = 1;
  for (size_t i = 0; i < index_size; ++i) {
    if (advance_current_rank(comparator, index, i)) {
      crt_rank = i + 1;
    }
    rank[index[i]] = crt_rank;
  }
  return rank;
}

// Computes the mapping from row position to dense rank.
std::vector<int64_t> index_to_dense_rank(
    const int64_t* index,
    const size_t index_size,
    const std::function<bool(const int64_t lhs, const int64_t rhs)>& comparator) {
  std::vector<int64_t> dense_rank(index_size);
  size_t crt_rank = 1;
  for (size_t i = 0; i < index_size; ++i) {
    if (advance_current_rank(comparator, index, i)) {
      ++crt_rank;
    }
    dense_rank[index[i]] = crt_rank;
  }
  return dense_rank;
}

// Computes the mapping from row position to percent rank.
std::vector<double> index_to_percent_rank(
    const int64_t* index,
    const size_t index_size,
    const std::function<bool(const int64_t lhs, const int64_t rhs)>& comparator) {
  std::vector<double> percent_rank(index_size);
  size_t crt_rank = 1;
  for (size_t i = 0; i < index_size; ++i) {
    if (advance_current_rank(comparator, index, i)) {
      crt_rank = i + 1;
    }
    percent_rank[index[i]] =
        index_size == 1 ? 0 : static_cast<double>(crt_rank - 1) / (index_size - 1);
  }
  return percent_rank;
}

// Computes the mapping from row position to cumulative distribution.
std::vector<double> index_to_cume_dist(
    const int64_t* index,
    const size_t index_size,
    const std::function<bool(const int64_t lhs, const int64_t rhs)>& comparator) {
  std::vector<double> cume_dist(index_size);
  size_t start_peer_group = 0;
  while (start_peer_group < index_size) {
    size_t end_peer_group = start_peer_group + 1;
    while (end_peer_group < index_size &&
           !advance_current_rank(comparator, index, end_peer_group)) {
      ++end_peer_group;
    }
    for (size_t i = start_peer_group; i < end_peer_group; ++i) {
      cume_dist[index[i]] = static_cast<double>(end_peer_group) / index_size;
    }
    start_peer_group = end_peer_group;
  }
  return cume_dist;
}

// Computes the mapping from row position to the n-tile statistic.
std::vector<int64_t> index_to_ntile(const int64_t* index,
                                    const size_t index_size,
                                    const size_t n) {
  std::vector<int64_t> row_numbers(index_size);
  if (!n) {
    throw std::runtime_error("NTILE argument cannot be zero");
  }
  const size_t tile_size = (index_size + n - 1) / n;
  for (size_t i = 0; i < index_size; ++i) {
    row_numbers[index[i]] = i / tile_size + 1;
  }
  return row_numbers;
}

// The element size in the result buffer for the given window function kind. Currently
// it's always 8.
size_t window_function_buffer_element_size(const SqlWindowFunctionKind /*kind*/) {
  return 8;
}

// Extracts the integer constant from a constant expression.
size_t get_int_constant_from_expr(const Analyzer::Expr* expr) {
  const auto lag_constant = dynamic_cast<const Analyzer::Constant*>(expr);
  if (!lag_constant) {
    throw std::runtime_error("LAG with non-constant lag argument not supported yet");
  }
  const auto& lag_ti = lag_constant->get_type_info();
  switch (lag_ti.get_type()) {
    case kSMALLINT: {
      return lag_constant->get_constval().smallintval;
    }
    case kINT: {
      return lag_constant->get_constval().intval;
    }
    case kBIGINT: {
      return lag_constant->get_constval().bigintval;
    }
    default: { LOG(FATAL) << "Invalid type for the lag argument"; }
  }
  return 0;
}

// Gets the lag or lead argument canonicalized as lag (lag = -lead).
ssize_t get_lag_or_lead_argument(const Analyzer::WindowFunction* window_func) {
  CHECK(window_func->getKind() == SqlWindowFunctionKind::LAG ||
        window_func->getKind() == SqlWindowFunctionKind::LEAD);
  const auto& args = window_func->getArgs();
  if (args.size() == 3) {
    throw std::runtime_error("LAG with default not supported yet");
  }
  if (args.size() == 2) {
    const ssize_t lag_or_lead = get_int_constant_from_expr(args[1].get());
    return window_func->getKind() == SqlWindowFunctionKind::LAG ? lag_or_lead
                                                                : -lag_or_lead;
  }
  CHECK_EQ(args.size(), size_t(1));
  return 1;
}

// Redistributes the original_indices according to the permutation given by
// output_for_partition_buff, reusing it as an output buffer.
void apply_permutation_to_partition(int64_t* output_for_partition_buff,
                                    const int32_t* original_indices,
                                    const size_t partition_size) {
  std::vector<int64_t> new_output_for_partition_buff(partition_size);
  for (size_t i = 0; i < partition_size; ++i) {
    new_output_for_partition_buff[i] = original_indices[output_for_partition_buff[i]];
  }
  std::copy(new_output_for_partition_buff.begin(),
            new_output_for_partition_buff.end(),
            output_for_partition_buff);
}

// Applies a lag to the given sorted_indices, reusing it as an output buffer.
void apply_lag_to_partition(const ssize_t lag,
                            const int32_t* original_indices,
                            int64_t* sorted_indices,
                            const size_t partition_size) {
  std::vector<int64_t> lag_sorted_indices(partition_size, -1);
  for (ssize_t idx = 0; idx < static_cast<ssize_t>(partition_size); ++idx) {
    ssize_t lag_idx = idx - lag;
    if (lag_idx < 0 || lag_idx >= static_cast<ssize_t>(partition_size)) {
      continue;
    }
    lag_sorted_indices[idx] = sorted_indices[lag_idx];
  }
  std::vector<int64_t> lag_original_indices(partition_size);
  for (size_t k = 0; k < partition_size; ++k) {
    const auto lag_index = lag_sorted_indices[k];
    lag_original_indices[sorted_indices[k]] =
        lag_index != -1 ? original_indices[lag_index] : -1;
  }
  std::copy(lag_original_indices.begin(), lag_original_indices.end(), sorted_indices);
}

// Computes first value function for the given output_for_partition_buff, reusing it as an
// output buffer.
void apply_first_value_to_partition(const int32_t* original_indices,
                                    int64_t* output_for_partition_buff,
                                    const size_t partition_size) {
  const auto first_value_idx = original_indices[output_for_partition_buff[0]];
  std::fill(output_for_partition_buff,
            output_for_partition_buff + partition_size,
            first_value_idx);
}

// Computes last value function for the given output_for_partition_buff, reusing it as an
// output buffer.
void apply_last_value_to_partition(const int32_t* original_indices,
                                   int64_t* output_for_partition_buff,
                                   const size_t partition_size) {
  std::copy(
      original_indices, original_indices + partition_size, output_for_partition_buff);
}

void index_to_partition_end(
    const int8_t* partition_end,
    const size_t off,
    const int64_t* index,
    const size_t index_size,
    const std::function<bool(const int64_t lhs, const int64_t rhs)>& comparator) {
  int64_t partition_end_handle = reinterpret_cast<int64_t>(partition_end);
  for (size_t i = 0; i < index_size; ++i) {
    if (advance_current_rank(comparator, index, i)) {
      agg_count_distinct_bitmap(&partition_end_handle, off + i - 1, 0);
    }
  }
  CHECK(index_size);
  agg_count_distinct_bitmap(&partition_end_handle, off + index_size - 1, 0);
}

bool pos_is_set(const int64_t bitset, const int64_t pos) {
  return (reinterpret_cast<const int8_t*>(bitset))[pos >> 3] & (1 << (pos & 7));
}

}  // namespace

// Write value to pending integer outputs collected for all the peer rows. The end of
// groups is represented by the bitset.
extern "C" void apply_window_pending_outputs_int(const int64_t handle,
                                                 const int64_t value,
                                                 const int64_t bitset,
                                                 const int64_t pos) {
  if (!pos_is_set(bitset, pos)) {
    return;
  }
  auto& pending_output_slots = *reinterpret_cast<std::vector<void*>*>(handle);
  for (auto pending_output_slot : pending_output_slots) {
    *reinterpret_cast<int64_t*>(pending_output_slot) = value;
  }
  pending_output_slots.clear();
}

extern "C" void apply_window_pending_outputs_double(const int64_t handle,
                                                    const double value,
                                                    const int64_t bitset,
                                                    const int64_t pos) {
  if (!pos_is_set(bitset, pos)) {
    return;
  }
  auto& pending_output_slots = *reinterpret_cast<std::vector<void*>*>(handle);
  for (auto pending_output_slot : pending_output_slots) {
    *reinterpret_cast<double*>(pending_output_slot) = value;
  }
  pending_output_slots.clear();
}

extern "C" void apply_window_pending_outputs_float(const int64_t handle,
                                                   const float value,
                                                   const int64_t bitset,
                                                   const int64_t pos) {
  if (!pos_is_set(bitset, pos)) {
    return;
  }
  auto& pending_output_slots = *reinterpret_cast<std::vector<void*>*>(handle);
  for (auto pending_output_slot : pending_output_slots) {
    *reinterpret_cast<double*>(pending_output_slot) = value;
  }
  pending_output_slots.clear();
}

// Add a pending output slot to be written back at the end of a peer row group.
extern "C" void add_window_pending_output(void* pending_output, const int64_t handle) {
  reinterpret_cast<std::vector<void*>*>(handle)->push_back(pending_output);
}

// Returns true for value window functions, false otherwise.
bool window_function_is_value(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::LAG:
    case SqlWindowFunctionKind::LEAD:
    case SqlWindowFunctionKind::FIRST_VALUE:
    case SqlWindowFunctionKind::LAST_VALUE: {
      return true;
    }
    default: { return false; }
  }
}

// Returns true for aggregate window functions, false otherwise.
bool window_function_is_aggregate(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::AVG:
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX:
    case SqlWindowFunctionKind::SUM:
    case SqlWindowFunctionKind::COUNT:
    case SqlWindowFunctionKind::SUM_INTERNAL: {
      return true;
    }
    default: { return false; }
  }
}

// Returns true iff the aggregate window function requires special multiplicity handling
// to ensure that peer rows have the same value for the window function.
bool window_function_requires_peer_handling(const Analyzer::WindowFunction* window_func) {
  if (!window_function_is_aggregate(window_func->getKind())) {
    return false;
  }
  if (window_func->getOrderKeys().empty()) {
    return true;
  }
  switch (window_func->getKind()) {
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX: {
      return false;
    }
    default: { return true; }
  }
}

void WindowFunctionContext::compute() {
  CHECK(!output_);
  output_ = static_cast<int8_t*>(checked_malloc(
      elem_count_ * window_function_buffer_element_size(window_func_->getKind())));
  if (window_function_is_aggregate(window_func_->getKind())) {
    fillPartitionStart();
    if (window_function_requires_peer_handling(window_func_)) {
      fillPartitionEnd();
    }
  }
  std::unique_ptr<int64_t[]> scratchpad(new int64_t[elem_count_]);
  int64_t off = 0;
  for (size_t i = 0; i < partitionCount(); ++i) {
    auto partition_size = counts()[i];
    if (partition_size == 0) {
      continue;
    }
    auto output_for_partition_buff = scratchpad.get() + offsets()[i];
    std::iota(output_for_partition_buff,
              output_for_partition_buff + partition_size,
              int64_t(0));
    std::vector<Comparator> comparators;
    const auto& order_keys = window_func_->getOrderKeys();
    const auto& collation = window_func_->getCollation();
    CHECK_EQ(order_keys.size(), collation.size());
    for (size_t order_column_idx = 0; order_column_idx < order_columns_.size();
         ++order_column_idx) {
      auto order_column_buffer = order_columns_[order_column_idx];
      const auto order_col =
          dynamic_cast<const Analyzer::ColumnVar*>(order_keys[order_column_idx].get());
      CHECK(order_col);
      const auto& order_col_collation = collation[order_column_idx];
      const auto asc_comparator = makeComparator(order_col,
                                                 order_column_buffer,
                                                 payload() + offsets()[i],
                                                 order_col_collation.nulls_first);
      auto comparator = asc_comparator;
      if (order_col_collation.is_desc) {
        comparator = [asc_comparator](const int64_t lhs, const int64_t rhs) {
          return asc_comparator(rhs, lhs);
        };
      }
      comparators.push_back(comparator);
    }
    const auto col_tuple_comparator = [&comparators](const int64_t lhs,
                                                     const int64_t rhs) {
      for (const auto& comparator : comparators) {
        if (comparator(lhs, rhs)) {
          return true;
        }
      }
      return false;
    };
    std::sort(output_for_partition_buff,
              output_for_partition_buff + partition_size,
              col_tuple_comparator);
    computePartition(output_for_partition_buff,
                     partition_size,
                     off,
                     window_func_,
                     col_tuple_comparator);
    if (window_function_is_value(window_func_->getKind()) ||
        window_function_is_aggregate(window_func_->getKind())) {
      off += partition_size;
    }
  }
  if (window_function_is_value(window_func_->getKind()) ||
      window_function_is_aggregate(window_func_->getKind())) {
    CHECK_EQ(off, elem_count_);
  }
  auto output_i64 = reinterpret_cast<int64_t*>(output_);
  if (window_function_is_aggregate(window_func_->getKind())) {
    std::copy(scratchpad.get(), scratchpad.get() + elem_count_, output_i64);
  } else {
    for (size_t i = 0; i < elem_count_; ++i) {
      output_i64[payload()[i]] = scratchpad[i];
    }
  }
}

const Analyzer::WindowFunction* WindowFunctionContext::getWindowFunction() const {
  return window_func_;
}

const int8_t* WindowFunctionContext::output() const {
  return output_;
}

const int64_t* WindowFunctionContext::aggregateState() const {
  CHECK(window_function_is_aggregate(window_func_->getKind()));
  return &aggregate_state_.val;
}

const int64_t* WindowFunctionContext::aggregateStateCount() const {
  CHECK(window_function_is_aggregate(window_func_->getKind()));
  return &aggregate_state_.count;
}

int64_t WindowFunctionContext::aggregateStatePendingOutputs() const {
  CHECK(window_function_is_aggregate(window_func_->getKind()));
  return reinterpret_cast<int64_t>(&aggregate_state_.outputs);
}

const int8_t* WindowFunctionContext::partitionStart() const {
  return partition_start_;
}

const int8_t* WindowFunctionContext::partitionEnd() const {
  return partition_end_;
}

size_t WindowFunctionContext::elementCount() const {
  return elem_count_;
}

void WindowFunctionContext::setRowNumber(llvm::Value* row_number) {
  aggregate_state_.row_number = row_number;
}

llvm::Value* WindowFunctionContext::getRowNumber() const {
  return aggregate_state_.row_number;
}

namespace {

template <class T>
bool integer_comparator(const int8_t* order_column_buffer,
                        const SQLTypeInfo& ti,
                        const int32_t* partition_indices,
                        const int64_t lhs,
                        const int64_t rhs,
                        const bool nulls_first) {
  const auto values = reinterpret_cast<const T*>(order_column_buffer);
  const auto lhs_val = values[partition_indices[lhs]];
  const auto rhs_val = values[partition_indices[rhs]];
  const auto null_val = inline_fixed_encoding_null_val(ti);
  if (lhs_val == null_val && rhs_val == null_val) {
    return false;
  }
  if (lhs_val == null_val && rhs_val != null_val) {
    return nulls_first;
  }
  if (rhs_val == null_val && lhs_val != null_val) {
    return !nulls_first;
  }
  return lhs_val < rhs_val;
}

template <class T, class NullPatternType>
bool fp_comparator(const int8_t* order_column_buffer,
                   const SQLTypeInfo& ti,
                   const int32_t* partition_indices,
                   const int64_t lhs,
                   const int64_t rhs,
                   const bool nulls_first) {
  const auto values = reinterpret_cast<const T*>(order_column_buffer);
  const auto lhs_val = values[partition_indices[lhs]];
  const auto rhs_val = values[partition_indices[rhs]];
  const auto null_bit_pattern = null_val_bit_pattern(ti, ti.get_type() == kFLOAT);
  const auto lhs_bit_pattern =
      *reinterpret_cast<const NullPatternType*>(may_alias_ptr(&lhs_val));
  const auto rhs_bit_pattern =
      *reinterpret_cast<const NullPatternType*>(may_alias_ptr(&rhs_val));
  if (lhs_bit_pattern == null_bit_pattern && rhs_bit_pattern == null_bit_pattern) {
    return false;
  }
  if (lhs_bit_pattern == null_bit_pattern && rhs_bit_pattern != null_bit_pattern) {
    return nulls_first;
  }
  if (rhs_bit_pattern == null_bit_pattern && lhs_bit_pattern != null_bit_pattern) {
    return !nulls_first;
  }
  return lhs_val < rhs_val;
}

}  // namespace

std::function<bool(const int64_t lhs, const int64_t rhs)>
WindowFunctionContext::makeComparator(const Analyzer::ColumnVar* col_var,
                                      const int8_t* order_column_buffer,
                                      const int32_t* partition_indices,
                                      const bool nulls_first) {
  const auto& ti = col_var->get_type_info();
  if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
    switch (ti.get_size()) {
      case 8: {
        return [order_column_buffer, nulls_first, partition_indices, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return integer_comparator<int64_t>(
              order_column_buffer, ti, partition_indices, lhs, rhs, nulls_first);
        };
      }
      case 4: {
        return [order_column_buffer, nulls_first, partition_indices, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return integer_comparator<int32_t>(
              order_column_buffer, ti, partition_indices, lhs, rhs, nulls_first);
        };
      }
      case 2: {
        return [order_column_buffer, nulls_first, partition_indices, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return integer_comparator<int16_t>(
              order_column_buffer, ti, partition_indices, lhs, rhs, nulls_first);
        };
      }
      case 1: {
        return [order_column_buffer, nulls_first, partition_indices, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return integer_comparator<int8_t>(
              order_column_buffer, ti, partition_indices, lhs, rhs, nulls_first);
        };
      }
      default: { LOG(FATAL) << "Invalid type size: " << ti.get_size(); }
    }
  }
  if (ti.is_fp()) {
    switch (ti.get_type()) {
      case kFLOAT: {
        return [order_column_buffer, nulls_first, partition_indices, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return fp_comparator<float, int32_t>(
              order_column_buffer, ti, partition_indices, lhs, rhs, nulls_first);
        };
      }
      case kDOUBLE: {
        return [order_column_buffer, nulls_first, partition_indices, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return fp_comparator<double, int64_t>(
              order_column_buffer, ti, partition_indices, lhs, rhs, nulls_first);
        };
      }
      default: { LOG(FATAL) << "Invalid float type"; }
    }
  }
  throw std::runtime_error("Type not supported yet");
}

void WindowFunctionContext::computePartition(
    int64_t* output_for_partition_buff,
    const size_t partition_size,
    const size_t off,
    const Analyzer::WindowFunction* window_func,
    const std::function<bool(const int64_t lhs, const int64_t rhs)>& comparator) {
  switch (window_func->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER: {
      const auto row_numbers =
          index_to_row_number(output_for_partition_buff, partition_size);
      std::copy(row_numbers.begin(), row_numbers.end(), output_for_partition_buff);
      break;
    }
    case SqlWindowFunctionKind::RANK: {
      const auto rank =
          index_to_rank(output_for_partition_buff, partition_size, comparator);
      std::copy(rank.begin(), rank.end(), output_for_partition_buff);
      break;
    }
    case SqlWindowFunctionKind::DENSE_RANK: {
      const auto dense_rank =
          index_to_dense_rank(output_for_partition_buff, partition_size, comparator);
      std::copy(dense_rank.begin(), dense_rank.end(), output_for_partition_buff);
      break;
    }
    case SqlWindowFunctionKind::PERCENT_RANK: {
      const auto percent_rank =
          index_to_percent_rank(output_for_partition_buff, partition_size, comparator);
      std::copy(percent_rank.begin(),
                percent_rank.end(),
                reinterpret_cast<double*>(may_alias_ptr(output_for_partition_buff)));
      break;
    }
    case SqlWindowFunctionKind::CUME_DIST: {
      const auto cume_dist =
          index_to_cume_dist(output_for_partition_buff, partition_size, comparator);
      std::copy(cume_dist.begin(),
                cume_dist.end(),
                reinterpret_cast<double*>(may_alias_ptr(output_for_partition_buff)));
      break;
    }
    case SqlWindowFunctionKind::NTILE: {
      const auto& args = window_func->getArgs();
      CHECK_EQ(args.size(), size_t(1));
      const auto n = get_int_constant_from_expr(args.front().get());
      const auto ntile = index_to_ntile(output_for_partition_buff, partition_size, n);
      std::copy(ntile.begin(), ntile.end(), output_for_partition_buff);
      break;
    }
    case SqlWindowFunctionKind::LAG:
    case SqlWindowFunctionKind::LEAD: {
      const auto lag_or_lead = get_lag_or_lead_argument(window_func);
      const auto partition_row_offsets = payload() + off;
      apply_lag_to_partition(
          lag_or_lead, partition_row_offsets, output_for_partition_buff, partition_size);
      break;
    }
    case SqlWindowFunctionKind::FIRST_VALUE: {
      const auto partition_row_offsets = payload() + off;
      apply_first_value_to_partition(
          partition_row_offsets, output_for_partition_buff, partition_size);
      break;
    }
    case SqlWindowFunctionKind::LAST_VALUE: {
      const auto partition_row_offsets = payload() + off;
      apply_last_value_to_partition(
          partition_row_offsets, output_for_partition_buff, partition_size);
      break;
    }
    case SqlWindowFunctionKind::AVG:
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX:
    case SqlWindowFunctionKind::SUM:
    case SqlWindowFunctionKind::COUNT: {
      const auto partition_row_offsets = payload() + off;
      if (window_function_requires_peer_handling(window_func)) {
        index_to_partition_end(
            partitionEnd(), off, output_for_partition_buff, partition_size, comparator);
      }
      apply_permutation_to_partition(
          output_for_partition_buff, partition_row_offsets, partition_size);
      break;
    }
    default: {
      throw std::runtime_error("Window function not supported yet: " +
                               sql_window_function_to_str(window_func->getKind()));
    }
  }
}

void WindowFunctionContext::fillPartitionStart() {
  CountDistinctDescriptor partition_start_bitmap{CountDistinctImplType::Bitmap,
                                                 0,
                                                 static_cast<int64_t>(elem_count_),
                                                 false,
                                                 ExecutorDeviceType::CPU,
                                                 1};
  partition_start_ = static_cast<int8_t*>(
      checked_calloc(partition_start_bitmap.bitmapPaddedSizeBytes(), 1));
  ssize_t partition_count = partitionCount();
  std::vector<size_t> partition_offsets(partition_count);
  std::partial_sum(counts(), counts() + partition_count, partition_offsets.begin());
  auto partition_start_handle = reinterpret_cast<int64_t>(partition_start_);
  agg_count_distinct_bitmap(&partition_start_handle, 0, 0);
  for (ssize_t i = 0; i < partition_count - 1; ++i) {
    agg_count_distinct_bitmap(&partition_start_handle, partition_offsets[i], 0);
  }
}

void WindowFunctionContext::fillPartitionEnd() {
  CountDistinctDescriptor partition_start_bitmap{CountDistinctImplType::Bitmap,
                                                 0,
                                                 static_cast<int64_t>(elem_count_),
                                                 false,
                                                 ExecutorDeviceType::CPU,
                                                 1};
  partition_end_ = static_cast<int8_t*>(
      checked_calloc(partition_start_bitmap.bitmapPaddedSizeBytes(), 1));
  ssize_t partition_count = partitionCount();
  std::vector<size_t> partition_offsets(partition_count);
  std::partial_sum(counts(), counts() + partition_count, partition_offsets.begin());
  auto partition_end_handle = reinterpret_cast<int64_t>(partition_end_);
  for (ssize_t i = 0; i < partition_count - 1; ++i) {
    if (partition_offsets[i] == 0) {
      continue;
    }
    agg_count_distinct_bitmap(&partition_end_handle, partition_offsets[i] - 1, 0);
  }
  agg_count_distinct_bitmap(&partition_end_handle, elem_count_ - 1, 0);
}

const int32_t* WindowFunctionContext::payload() const {
  return reinterpret_cast<const int32_t*>(
      partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->payloadBufferOff());
}

const int32_t* WindowFunctionContext::offsets() const {
  return reinterpret_cast<const int32_t*>(
      partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->offsetBufferOff());
}

const int32_t* WindowFunctionContext::counts() const {
  return reinterpret_cast<const int32_t*>(
      partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->countBufferOff());
}

size_t WindowFunctionContext::partitionCount() const {
  const auto partition_count = counts() - offsets();
  CHECK_GE(partition_count, 0);
  return partition_count;
}

void WindowProjectNodeContext::addWindowFunctionContext(
    std::unique_ptr<WindowFunctionContext> window_function_context,
    const size_t target_index) {
  const auto it_ok = window_contexts_.emplace(
      std::make_pair(target_index, std::move(window_function_context)));
  CHECK(it_ok.second);
}

const WindowFunctionContext* WindowProjectNodeContext::activateWindowFunctionContext(
    const size_t target_index) const {
  const auto it = window_contexts_.find(target_index);
  CHECK(it != window_contexts_.end());
  s_active_window_function_ = it->second.get();
  return s_active_window_function_;
}

void WindowProjectNodeContext::resetWindowFunctionContext() {
  s_active_window_function_ = nullptr;
}

WindowFunctionContext* WindowProjectNodeContext::getActiveWindowFunctionContext() {
  return s_active_window_function_;
}

WindowProjectNodeContext* WindowProjectNodeContext::create() {
  s_instance_ = std::make_unique<WindowProjectNodeContext>();
  return s_instance_.get();
}

const WindowProjectNodeContext* WindowProjectNodeContext::get() {
  return s_instance_.get();
}

void WindowProjectNodeContext::reset() {
  s_instance_ = nullptr;
  s_active_window_function_ = nullptr;
}

std::unique_ptr<WindowProjectNodeContext> WindowProjectNodeContext::s_instance_;
WindowFunctionContext* WindowProjectNodeContext::s_active_window_function_{nullptr};
