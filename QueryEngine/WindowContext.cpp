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

#include "QueryEngine/WindowContext.h"

#include <numeric>

#include "QueryEngine/Descriptors/CountDistinctDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/ResultSetBufferAccessors.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/TypePunning.h"
#include "QueryEngine/Utils/SegmentTree.h"
#include "Shared/Intervals.h"
#include "Shared/checked_alloc.h"
#include "Shared/funcannotations.h"
#include "Shared/sqltypes.h"
#include "Shared/threading.h"

#ifdef HAVE_TBB
//#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#else
#include <thrust/sort.h>
#endif

bool g_enable_parallel_window_partition_compute{true};
size_t g_parallel_window_partition_compute_threshold{1 << 12};  // 4096

bool g_enable_parallel_window_partition_sort{true};
size_t g_parallel_window_partition_sort_threshold{1 << 10};  // 1024

size_t g_window_function_aggregation_tree_fanout{8};

// Non-partitioned version (no hash table provided)
WindowFunctionContext::WindowFunctionContext(
    const Analyzer::WindowFunction* window_func,
    const size_t elem_count,
    const ExecutorDeviceType device_type,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : window_func_(window_func)
    , partition_cache_key_(EMPTY_HASHED_PLAN_DAG_KEY)
    , sorted_partition_cache_key_(EMPTY_HASHED_PLAN_DAG_KEY)
    , partitions_(nullptr)
    , elem_count_(elem_count)
    , output_(nullptr)
    , sorted_partition_buf_(nullptr)
    , aggregate_trees_fan_out_(g_window_function_aggregation_tree_fanout)
    , aggregate_trees_depth_(nullptr)
    , aggregate_tree_null_start_pos_(nullptr)
    , aggregate_tree_null_end_pos_(nullptr)
    , partition_start_offset_(nullptr)
    , partition_start_(nullptr)
    , partition_end_(nullptr)
    , device_type_(device_type)
    , row_set_mem_owner_(row_set_mem_owner)
    , dummy_count_(elem_count)
    , dummy_offset_(0)
    , dummy_payload_(nullptr) {
  CHECK_LE(elem_count_, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  if (elem_count_ > 0) {
    dummy_payload_ =
        reinterpret_cast<int32_t*>(checked_malloc(elem_count_ * sizeof(int32_t)));
    std::iota(dummy_payload_, dummy_payload_ + elem_count_, int32_t(0));
    if (window_func_->hasFraming()) {
      // in this case, we consider all rows of the row belong to the same and only
      // existing partition
      partition_start_offset_ =
          reinterpret_cast<int64_t*>(checked_calloc(2, sizeof(int64_t)));
      partition_start_offset_[1] = elem_count_;
      aggregate_trees_depth_ =
          reinterpret_cast<size_t*>(checked_calloc(1, sizeof(size_t)));
      aggregate_tree_null_start_pos_ =
          reinterpret_cast<int64_t*>(checked_calloc(1, sizeof(int64_t)));
      aggregate_tree_null_end_pos_ =
          reinterpret_cast<int64_t*>(checked_calloc(1, sizeof(int64_t)));
    }
  }
}

// Partitioned version
WindowFunctionContext::WindowFunctionContext(
    const Analyzer::WindowFunction* window_func,
    QueryPlanHash partition_cache_key,
    const std::shared_ptr<HashJoin>& partitions,
    const size_t elem_count,
    const ExecutorDeviceType device_type,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    size_t aggregation_tree_fan_out)
    : window_func_(window_func)
    , partition_cache_key_(partition_cache_key)
    , sorted_partition_cache_key_(EMPTY_HASHED_PLAN_DAG_KEY)
    , partitions_(partitions)
    , elem_count_(elem_count)
    , output_(nullptr)
    , sorted_partition_buf_(nullptr)
    , aggregate_trees_fan_out_(aggregation_tree_fan_out)
    , aggregate_trees_depth_(nullptr)
    , aggregate_tree_null_start_pos_(nullptr)
    , aggregate_tree_null_end_pos_(nullptr)
    , partition_start_offset_(nullptr)
    , partition_start_(nullptr)
    , partition_end_(nullptr)
    , device_type_(device_type)
    , row_set_mem_owner_(row_set_mem_owner)
    , dummy_count_(elem_count)
    , dummy_offset_(0)
    , dummy_payload_(nullptr) {
  CHECK(partitions_);  // This version should have hash table
  size_t partition_count = partitionCount();
  partition_start_offset_ =
      reinterpret_cast<int64_t*>(checked_calloc(partition_count + 1, sizeof(int64_t)));
  if (window_func_->hasFraming()) {
    aggregate_trees_depth_ =
        reinterpret_cast<size_t*>(checked_calloc(partition_count, sizeof(size_t)));
    aggregate_tree_null_start_pos_ =
        reinterpret_cast<int64_t*>(checked_calloc(partition_count, sizeof(int64_t)));
    aggregate_tree_null_end_pos_ =
        reinterpret_cast<int64_t*>(checked_calloc(partition_count, sizeof(int64_t)));
  }
  // the first partition starts at zero position
  std::partial_sum(counts(), counts() + partition_count, partition_start_offset_ + 1);
}

WindowFunctionContext::~WindowFunctionContext() {
  free(partition_start_);
  free(partition_end_);
  if (dummy_payload_) {
    free(dummy_payload_);
  }
  if (partition_start_offset_) {
    free(partition_start_offset_);
  }
  if (aggregate_trees_depth_) {
    free(aggregate_trees_depth_);
  }
  if (aggregate_tree_null_start_pos_) {
    free(aggregate_tree_null_start_pos_);
  }
  if (aggregate_tree_null_end_pos_) {
    free(aggregate_tree_null_end_pos_);
  }
}

void WindowFunctionContext::addOrderColumn(
    const int8_t* column,
    const SQLTypeInfo& ti,
    const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner) {
  order_columns_owner_.push_back(chunks_owner);
  order_columns_.push_back(column);
  order_columns_ti_.push_back(ti);
}

void WindowFunctionContext::addColumnBufferForWindowFunctionExpression(
    const int8_t* column,
    const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner) {
  window_func_expr_columns_owner_.push_back(chunks_owner);
  window_func_expr_columns_.push_back(column);
};

const std::vector<const int8_t*>& WindowFunctionContext::getOrderKeyColumnBuffers()
    const {
  return order_columns_;
}

const std::vector<SQLTypeInfo>& WindowFunctionContext::getOrderKeyColumnBufferTypes()
    const {
  return order_columns_ti_;
}

void WindowFunctionContext::setSortedPartitionCacheKey(QueryPlanHash cache_key) {
  sorted_partition_cache_key_ = cache_key;
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
    default: {
      LOG(FATAL) << "Invalid type for the lag argument";
    }
  }
  return 0;
}

// Gets the lag or lead argument canonicalized as lag (lag = -lead).
int64_t get_lag_or_lead_argument(const Analyzer::WindowFunction* window_func) {
  CHECK(window_func->getKind() == SqlWindowFunctionKind::LAG ||
        window_func->getKind() == SqlWindowFunctionKind::LEAD);
  const auto& args = window_func->getArgs();
  if (args.size() == 3) {
    throw std::runtime_error("LAG with default not supported yet");
  }
  if (args.size() == 2) {
    const int64_t lag_or_lead =
        static_cast<int64_t>(get_int_constant_from_expr(args[1].get()));
    return window_func->getKind() == SqlWindowFunctionKind::LAG ? lag_or_lead
                                                                : -lag_or_lead;
  }
  CHECK_EQ(args.size(), size_t(1));
  return window_func->getKind() == SqlWindowFunctionKind::LAG ? 1 : -1;
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
void apply_lag_to_partition(const int64_t lag,
                            const int32_t* original_indices,
                            int64_t* sorted_indices,
                            const size_t partition_size) {
  std::vector<int64_t> lag_sorted_indices(partition_size, -1);
  for (int64_t idx = 0; idx < static_cast<int64_t>(partition_size); ++idx) {
    int64_t lag_idx = idx - lag;
    if (lag_idx < 0 || lag_idx >= static_cast<int64_t>(partition_size)) {
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

// Write value to pending integer outputs collected for all the peer rows. The end of
// groups is represented by the bitset.
template <class T>
void apply_window_pending_outputs_int(const int64_t handle,
                                      const int64_t value,
                                      const int64_t bitset,
                                      const int64_t pos) {
  if (!pos_is_set(bitset, pos)) {
    return;
  }
  auto& pending_output_slots = *reinterpret_cast<std::vector<void*>*>(handle);
  for (auto pending_output_slot : pending_output_slots) {
    *reinterpret_cast<T*>(pending_output_slot) = value;
  }
  pending_output_slots.clear();
}

}  // namespace

extern "C" RUNTIME_EXPORT void apply_window_pending_outputs_int64(const int64_t handle,
                                                                  const int64_t value,
                                                                  const int64_t bitset,
                                                                  const int64_t pos) {
  apply_window_pending_outputs_int<int64_t>(handle, value, bitset, pos);
}

extern "C" RUNTIME_EXPORT void apply_window_pending_outputs_int32(const int64_t handle,
                                                                  const int64_t value,
                                                                  const int64_t bitset,
                                                                  const int64_t pos) {
  apply_window_pending_outputs_int<int32_t>(handle, value, bitset, pos);
}

extern "C" RUNTIME_EXPORT void apply_window_pending_outputs_int16(const int64_t handle,
                                                                  const int64_t value,
                                                                  const int64_t bitset,
                                                                  const int64_t pos) {
  apply_window_pending_outputs_int<int16_t>(handle, value, bitset, pos);
}

extern "C" RUNTIME_EXPORT void apply_window_pending_outputs_int8(const int64_t handle,
                                                                 const int64_t value,
                                                                 const int64_t bitset,
                                                                 const int64_t pos) {
  apply_window_pending_outputs_int<int8_t>(handle, value, bitset, pos);
}

extern "C" RUNTIME_EXPORT void apply_window_pending_outputs_double(const int64_t handle,
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

extern "C" RUNTIME_EXPORT void apply_window_pending_outputs_float(const int64_t handle,
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

extern "C" RUNTIME_EXPORT void apply_window_pending_outputs_float_columnar(
    const int64_t handle,
    const float value,
    const int64_t bitset,
    const int64_t pos) {
  if (!pos_is_set(bitset, pos)) {
    return;
  }
  auto& pending_output_slots = *reinterpret_cast<std::vector<void*>*>(handle);
  for (auto pending_output_slot : pending_output_slots) {
    *reinterpret_cast<float*>(pending_output_slot) = value;
  }
  pending_output_slots.clear();
}

// Add a pending output slot to be written back at the end of a peer row group.
extern "C" RUNTIME_EXPORT void add_window_pending_output(void* pending_output,
                                                         const int64_t handle) {
  reinterpret_cast<std::vector<void*>*>(handle)->push_back(pending_output);
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
    default: {
      return true;
    }
  }
}

void WindowFunctionContext::compute(
    std::unordered_map<QueryPlanHash, size_t>& sorted_partition_key_ref_count_map,
    std::unordered_map<QueryPlanHash, std::shared_ptr<std::vector<int64_t>>>&
        sorted_partition_cache) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(!output_);
  size_t output_buf_sz =
      elem_count_ * window_function_buffer_element_size(window_func_->getKind());
  output_ = static_cast<int8_t*>(row_set_mem_owner_->allocate(output_buf_sz,
                                                              /*thread_idx=*/0));
  const bool is_window_function_aggregate =
      window_function_is_aggregate(window_func_->getKind());
  if (is_window_function_aggregate) {
    fillPartitionStart();
    if (window_function_requires_peer_handling(window_func_)) {
      fillPartitionEnd();
    }
  }
  std::unique_ptr<int64_t[]> scratchpad;
  int64_t* intermediate_output_buffer;
  if (is_window_function_aggregate) {
    intermediate_output_buffer = reinterpret_cast<int64_t*>(output_);
  } else {
    output_buf_sz = sizeof(int64_t) * elem_count_;
    scratchpad.reset(new int64_t[elem_count_]);
    intermediate_output_buffer = scratchpad.get();
  }
  const bool should_parallelize{g_enable_parallel_window_partition_compute &&
                                elem_count_ >=
                                    g_parallel_window_partition_compute_threshold};

  auto cached_sorted_partition_it =
      sorted_partition_cache.find(sorted_partition_cache_key_);
  if (cached_sorted_partition_it != sorted_partition_cache.end()) {
    auto& sorted_partition = cached_sorted_partition_it->second;
    VLOG(1) << "Reuse cached sorted partition to compute window function context (key: "
            << sorted_partition_cache_key_
            << ", ordering condition: " << ::toString(window_func_->getOrderKeys())
            << ")";
    DEBUG_TIMER("Window Function Cached Sorted Partition Copy");
    std::memcpy(intermediate_output_buffer, sorted_partition->data(), output_buf_sz);
  } else {
    // ordering partitions if necessary
    const auto sort_partitions = [&](const size_t start, const size_t end) {
      for (size_t partition_idx = start; partition_idx < end; ++partition_idx) {
        sortPartition(partition_idx,
                      intermediate_output_buffer + offsets()[partition_idx],
                      should_parallelize);
      }
    };

    if (should_parallelize) {
      auto sorted_partition_copy_timer =
          DEBUG_TIMER("Window Function Partition Sorting Parallelized");
      threading::task_group thread_pool;
      for (auto interval : makeIntervals<size_t>(0, partitionCount(), cpu_threads())) {
        thread_pool.run([=] { sort_partitions(interval.begin, interval.end); });
      }
      thread_pool.wait();
    } else {
      auto sorted_partition_copy_timer =
          DEBUG_TIMER("Window Function  Partition Sorting Non-Parallelized");
      sort_partitions(0, partitionCount());
    }
    auto sorted_partition_ref_cnt_it =
        sorted_partition_key_ref_count_map.find(sorted_partition_cache_key_);
    bool can_access_sorted_partition =
        sorted_partition_ref_cnt_it != sorted_partition_key_ref_count_map.end() &&
        sorted_partition_ref_cnt_it->second > 1;
    bool has_framing_clause = window_func_->hasFraming();
    if (can_access_sorted_partition || has_framing_clause) {
      // keep the sorted partition only if it will be reused from other window function
      // context of this query
      sorted_partition_buf_ = std::make_shared<std::vector<int64_t>>(elem_count_);
      DEBUG_TIMER("Window Function Sorted Partition Copy For Caching");
      std::memcpy(
          sorted_partition_buf_->data(), intermediate_output_buffer, output_buf_sz);
      auto it = sorted_partition_cache.emplace(sorted_partition_cache_key_,
                                               sorted_partition_buf_);
      if (it.second) {
        VLOG(1) << "Put sorted partition to cache (key: " << sorted_partition_cache_key_
                << ", ordering condition: " << ::toString(window_func_->getOrderKeys())
                << ")";
      }
    }
  }

  if (window_func_->hasFraming()) {
    // let's allow aggregation functions first
    // todo (yoonmin) : support navigation functions, i.e., first_expr, last_expr, and
    // nth_expr
    CHECK(window_function_is_aggregate(window_func_->getKind()));
    // construct segment tree per partition to deal with aggregation over frame
    const auto build_aggregation_tree_for_partitions = [&](const size_t start,
                                                           const size_t end) {
      for (size_t partition_idx = start; partition_idx < end; ++partition_idx) {
        // build a segment tree for the partition
        // todo (yoonmin) : support generic window function expression
        // i.e., when window_func_expr_columns_.size() > 1
        buildAggregationTreeForPartition(
            window_func_->getKind(),
            partition_idx,
            counts()[partition_idx],
            window_func_expr_columns_.front(),
            payload() + offsets()[partition_idx],
            intermediate_output_buffer,
            window_func_->getArgs().front()->get_type_info());
      }
    };
    auto partition_count = partitionCount();
    if (should_parallelize) {
      auto partition_compuation_timer =
          DEBUG_TIMER("Window Function Build Segment Tree for Partitions");
      threading::task_group thread_pool;
      for (auto interval : makeIntervals<size_t>(0, partition_count, cpu_threads())) {
        thread_pool.run(
            [=] { build_aggregation_tree_for_partitions(interval.begin, interval.end); });
      }
      thread_pool.wait();
    } else {
      auto partition_compuation_timer =
          DEBUG_TIMER("Window Function Build Segment Tree for Partitions");
      build_aggregation_tree_for_partitions(0, partition_count);
    }
  }

  const auto compute_partitions = [&](const size_t start, const size_t end) {
    for (size_t partition_idx = start; partition_idx < end; ++partition_idx) {
      computePartitionBuffer(partition_idx,
                             intermediate_output_buffer + offsets()[partition_idx],
                             window_func_);
    }
  };

  if (should_parallelize) {
    auto partition_compuation_timer = DEBUG_TIMER("Window Function Partition Compute");
    threading::task_group thread_pool;
    for (auto interval : makeIntervals<size_t>(0, partitionCount(), cpu_threads())) {
      thread_pool.run([=] { compute_partitions(interval.begin, interval.end); });
    }
    thread_pool.wait();
  } else {
    auto partition_compuation_timer =
        DEBUG_TIMER("Window Function Non-Parallelized Partition Compute");
    compute_partitions(0, partitionCount());
  }

  if (is_window_function_aggregate) {
    // If window function is aggregate we were able to write to the final output buffer
    // directly in computePartition and we are done.
    return;
  }

  auto output_i64 = reinterpret_cast<int64_t*>(output_);
  const auto payload_copy = [&](const size_t start, const size_t end) {
    for (size_t i = start; i < end; ++i) {
      output_i64[payload()[i]] = intermediate_output_buffer[i];
    }
  };
  if (should_parallelize) {
    auto payload_copy_timer =
        DEBUG_TIMER("Window Function Non-Aggregate Payload Copy Parallelized");
    threading::task_group thread_pool;
    for (auto interval : makeIntervals<size_t>(
             0,
             elem_count_,
             std::min(static_cast<size_t>(cpu_threads()),
                      (elem_count_ + g_parallel_window_partition_compute_threshold - 1) /
                          g_parallel_window_partition_compute_threshold))) {
      thread_pool.run([=] { payload_copy(interval.begin, interval.end); });
    }
    thread_pool.wait();
  } else {
    auto payload_copy_timer =
        DEBUG_TIMER("Window Function Non-Aggregate Payload Copy Non-Parallelized");
    payload_copy(0, elem_count_);
  }
}

IndexPair WindowFunctionContext::computeNullRangeOfSortedPartition(
    SQLTypeInfo order_col_ti,
    size_t partition_idx,
    const int64_t* ordered_col_idx_buf) {
  IndexPair null_range;
  null_range.first = std::numeric_limits<int64_t>::max();
  null_range.second = std::numeric_limits<int64_t>::min();
  const auto& collation = window_func_->getCollation().front();
  const auto partition_size = counts()[partition_idx];
  if (partition_size > 0 && (order_col_ti.is_integer() || order_col_ti.is_decimal() ||
                             order_col_ti.is_time() || order_col_ti.is_boolean())) {
    const auto null_val = inline_int_null_val(order_col_ti);
    switch (order_col_ti.get_size()) {
      case 8: {
        const auto order_col_buf =
            reinterpret_cast<const int64_t*>(order_columns_.front()) +
            offsets()[partition_idx];
        if (collation.nulls_first && order_col_buf[ordered_col_idx_buf[0]] == null_val) {
          int64_t null_range_max = 1;
          while (order_col_buf[null_range_max] == null_val) {
            null_range_max++;
          }
          null_range.first = 0;
          null_range.second = null_range_max - 1;
        } else if (!collation.nulls_first &&
                   order_col_buf[ordered_col_idx_buf[partition_size - 1]] == null_val) {
          int64_t null_range_min = partition_size - 2;
          while (order_col_buf[ordered_col_idx_buf[null_range_min]] == null_val) {
            null_range_min--;
          }
          null_range.first = null_range_min + 1;
          null_range.second = partition_size - 1;
        }
        break;
      }
      case 4: {
        const auto order_col_buf =
            reinterpret_cast<const int32_t*>(order_columns_.front()) +
            offsets()[partition_idx];
        if (collation.nulls_first && order_col_buf[ordered_col_idx_buf[0]] == null_val) {
          int64_t null_range_max = 1;
          while (order_col_buf[ordered_col_idx_buf[null_range_max]] == null_val) {
            null_range_max++;
          }
          null_range.first = 0;
          null_range.second = null_range_max - 1;
        } else if (!collation.nulls_first &&
                   order_col_buf[ordered_col_idx_buf[partition_size - 1]] == null_val) {
          int64_t null_range_min = partition_size - 2;
          while (order_col_buf[ordered_col_idx_buf[null_range_min]] == null_val) {
            null_range_min--;
          }
          null_range.first = null_range_min + 1;
          null_range.second = partition_size - 1;
        }
        break;
      }
      case 2: {
        const auto order_col_buf =
            reinterpret_cast<const int16_t*>(order_columns_.front()) +
            offsets()[partition_idx];
        if (collation.nulls_first && order_col_buf[ordered_col_idx_buf[0]] == null_val) {
          int64_t null_range_max = 1;
          while (order_col_buf[ordered_col_idx_buf[null_range_max]] == null_val) {
            null_range_max++;
          }
          null_range.first = 0;
          null_range.second = null_range_max - 1;
        } else if (!collation.nulls_first &&
                   order_col_buf[ordered_col_idx_buf[partition_size - 1]] == null_val) {
          int64_t null_range_min = partition_size - 2;
          while (order_col_buf[ordered_col_idx_buf[null_range_min]] == null_val) {
            null_range_min--;
          }
          null_range.first = null_range_min + 1;
          null_range.second = partition_size - 1;
        }
        break;
      }
      case 1: {
        const auto order_col_buf =
            reinterpret_cast<const int8_t*>(order_columns_.front()) +
            offsets()[partition_idx];
        if (collation.nulls_first && order_col_buf[ordered_col_idx_buf[0]] == null_val) {
          int64_t null_range_max = 1;
          while (order_col_buf[ordered_col_idx_buf[null_range_max]] == null_val) {
            null_range_max++;
          }
          null_range.first = 0;
          null_range.second = null_range_max - 1;
        } else if (!collation.nulls_first &&
                   order_col_buf[ordered_col_idx_buf[partition_size - 1]] == null_val) {
          int64_t null_range_min = partition_size - 2;
          while (order_col_buf[ordered_col_idx_buf[null_range_min]] == null_val) {
            null_range_min--;
          }
          null_range.first = null_range_min + 1;
          null_range.second = partition_size - 1;
        }
        break;
      }
      default: {
        LOG(FATAL) << "Invalid type size: " << order_col_ti.get_size();
      }
    }
  }
  if (partition_size > 0 && order_col_ti.is_fp()) {
    const auto null_bit_pattern =
        null_val_bit_pattern(order_col_ti, order_col_ti.get_type() == kFLOAT);
    switch (order_col_ti.get_type()) {
      case kFLOAT: {
        const auto order_col_buf =
            reinterpret_cast<const float*>(order_columns_.front()) +
            offsets()[partition_idx];
        if (collation.nulls_first &&
            *reinterpret_cast<const int32_t*>(may_alias_ptr(
                &order_col_buf[ordered_col_idx_buf[0]])) == null_bit_pattern) {
          int64_t null_range_max = 1;
          while (*reinterpret_cast<const int32_t*>(may_alias_ptr(
                     &order_col_buf[ordered_col_idx_buf[null_range_max]])) ==
                 null_bit_pattern) {
            null_range_max++;
          }
          null_range.first = 0;
          null_range.second = null_range_max - 1;
        } else if (!collation.nulls_first &&
                   *reinterpret_cast<const int32_t*>(may_alias_ptr(
                       &order_col_buf[ordered_col_idx_buf[partition_size - 1]])) ==
                       null_bit_pattern) {
          int64_t null_range_min = partition_size - 2;
          while (*reinterpret_cast<const int32_t*>(may_alias_ptr(
                     &order_col_buf[ordered_col_idx_buf[null_range_min]])) ==
                 null_bit_pattern) {
            null_range_min--;
          }
          null_range.first = null_range_min + 1;
          null_range.second = partition_size - 1;
        }
        break;
      }
      case kDOUBLE: {
        const auto order_col_buf =
            reinterpret_cast<const double*>(order_columns_.front()) +
            offsets()[partition_idx];
        if (collation.nulls_first &&
            *reinterpret_cast<const int64_t*>(may_alias_ptr(
                &order_col_buf[ordered_col_idx_buf[0]])) == null_bit_pattern) {
          int64_t null_range_max = 1;
          while (*reinterpret_cast<const int64_t*>(may_alias_ptr(
                     &order_col_buf[ordered_col_idx_buf[null_range_max]])) ==
                 null_bit_pattern) {
            null_range_max++;
          }
          null_range.first = 0;
          null_range.second = null_range_max - 1;
        } else if (!collation.nulls_first &&
                   *reinterpret_cast<const int64_t*>(may_alias_ptr(
                       &order_col_buf[ordered_col_idx_buf[partition_size - 1]])) ==
                       null_bit_pattern) {
          int64_t null_range_min = partition_size - 2;
          while (*reinterpret_cast<const int64_t*>(may_alias_ptr(
                     &order_col_buf[ordered_col_idx_buf[null_range_min]])) ==
                 null_bit_pattern) {
            null_range_min--;
          }
          null_range.first = null_range_min + 1;
          null_range.second = partition_size - 1;
        }
        break;
      }
      default: {
        LOG(FATAL) << "Invalid float type";
      }
    }
  }
  return null_range;
}

std::vector<WindowFunctionContext::Comparator> WindowFunctionContext::createComparator(
    size_t partition_idx) {
  // create tuple comparator
  std::vector<WindowFunctionContext::Comparator> partition_comparator;
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
                                               payload() + offsets()[partition_idx],
                                               order_col_collation.nulls_first);
    auto comparator = asc_comparator;
    if (order_col_collation.is_desc) {
      comparator = [asc_comparator](const int64_t lhs, const int64_t rhs) {
        return asc_comparator(rhs, lhs);
      };
    }
    partition_comparator.push_back(comparator);
  }
  return partition_comparator;
}

void WindowFunctionContext::sortPartition(const size_t partition_idx,
                                          int64_t* output_for_partition_buff,
                                          bool should_parallelize) {
  const size_t partition_size{static_cast<size_t>(counts()[partition_idx])};
  if (partition_size == 0) {
    return;
  }
  std::iota(
      output_for_partition_buff, output_for_partition_buff + partition_size, int64_t(0));
  auto partition_comparator = createComparator(partition_idx);
  if (!partition_comparator.empty()) {
    const auto col_tuple_comparator = [&partition_comparator](const int64_t lhs,
                                                              const int64_t rhs) {
      for (const auto& comparator : partition_comparator) {
        const auto comparator_result = comparator(lhs, rhs);
        switch (comparator_result) {
          case WindowFunctionContext::WindowComparatorResult::LT:
            return true;
          case WindowFunctionContext::WindowComparatorResult::GT:
            return false;
          default:
            // WindowComparatorResult::EQ: continue to next comparator
            continue;
        }
      }
      // If here WindowFunctionContext::WindowComparatorResult::KEQ for all keys
      // return false as sort algo must enforce weak ordering
      return false;
    };
    if (should_parallelize) {
#ifdef HAVE_TBB
      tbb::parallel_sort(output_for_partition_buff,
                         output_for_partition_buff + partition_size,
                         col_tuple_comparator);
#else
      thrust::sort(output_for_partition_buff,
                   output_for_partition_buff + partition_size,
                   col_tuple_comparator);
#endif
    } else {
      std::sort(output_for_partition_buff,
                output_for_partition_buff + partition_size,
                col_tuple_comparator);
    }
  }
}

const Analyzer::WindowFunction* WindowFunctionContext::getWindowFunction() const {
  return window_func_;
}

const int8_t* WindowFunctionContext::output() const {
  return output_;
}

const int64_t* WindowFunctionContext::sortedPartition() const {
  CHECK(sorted_partition_buf_);
  return sorted_partition_buf_->data();
}

const int64_t* WindowFunctionContext::aggregateState() const {
  CHECK(window_function_is_aggregate(window_func_->getKind()));
  return &aggregate_state_.val;
}

const int64_t* WindowFunctionContext::aggregateStateCount() const {
  CHECK(window_function_is_aggregate(window_func_->getKind()));
  return &aggregate_state_.count;
}

const int64_t* WindowFunctionContext::partitionStartOffset() const {
  CHECK(partition_start_offset_);
  return partition_start_offset_;
}

const int64_t* WindowFunctionContext::partitionNumCountBuf() const {
  CHECK(partition_start_offset_);
  return partition_start_offset_ + 1;
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

namespace {

template <class T>
WindowFunctionContext::WindowComparatorResult integer_comparator(
    const int8_t* order_column_buffer,
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
    return WindowFunctionContext::WindowComparatorResult::EQ;
  }
  if (lhs_val == null_val && rhs_val != null_val) {
    return nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
                       : WindowFunctionContext::WindowComparatorResult::GT;
  }
  if (rhs_val == null_val && lhs_val != null_val) {
    return !nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
                        : WindowFunctionContext::WindowComparatorResult::GT;
  }
  if (lhs_val < rhs_val) {
    return WindowFunctionContext::WindowComparatorResult::LT;
  }
  if (lhs_val > rhs_val) {
    return WindowFunctionContext::WindowComparatorResult::GT;
  }
  return WindowFunctionContext::WindowComparatorResult::EQ;
}

template <class T, class NullPatternType>
WindowFunctionContext::WindowComparatorResult fp_comparator(
    const int8_t* order_column_buffer,
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
    return WindowFunctionContext::WindowComparatorResult::EQ;
  }
  if (lhs_bit_pattern == null_bit_pattern && rhs_bit_pattern != null_bit_pattern) {
    return nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
                       : WindowFunctionContext::WindowComparatorResult::GT;
  }
  if (rhs_bit_pattern == null_bit_pattern && lhs_bit_pattern != null_bit_pattern) {
    return !nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
                        : WindowFunctionContext::WindowComparatorResult::GT;
  }
  if (lhs_val < rhs_val) {
    return WindowFunctionContext::WindowComparatorResult::LT;
  }
  if (lhs_val > rhs_val) {
    return WindowFunctionContext::WindowComparatorResult::GT;
  }
  return WindowFunctionContext::WindowComparatorResult::EQ;
}

}  // namespace

WindowFunctionContext::Comparator WindowFunctionContext::makeComparator(
    const Analyzer::ColumnVar* col_var,
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
      default: {
        LOG(FATAL) << "Invalid type size: " << ti.get_size();
      }
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
      default: {
        LOG(FATAL) << "Invalid float type";
      }
    }
  }
  throw std::runtime_error("Type not supported yet");
}

void WindowFunctionContext::computePartitionBuffer(
    const size_t partition_idx,
    int64_t* output_for_partition_buff,
    const Analyzer::WindowFunction* window_func) {
  const size_t partition_size{static_cast<size_t>(counts()[partition_idx])};
  if (partition_size == 0) {
    return;
  }
  const auto offset = offsets()[partition_idx];
  auto partition_comparator = createComparator(partition_idx);
  const auto col_tuple_comparator = [&partition_comparator](const int64_t lhs,
                                                            const int64_t rhs) {
    for (const auto& comparator : partition_comparator) {
      const auto comparator_result = comparator(lhs, rhs);
      switch (comparator_result) {
        case WindowFunctionContext::WindowComparatorResult::LT:
          return true;
        case WindowFunctionContext::WindowComparatorResult::GT:
          return false;
        default:
          // WindowComparatorResult::EQ: continue to next comparator
          continue;
      }
    }
    // If here WindowFunctionContext::WindowComparatorResult::KEQ for all keys
    // return false as sort algo must enforce weak ordering
    return false;
  };
  switch (window_func->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER: {
      const auto row_numbers =
          index_to_row_number(output_for_partition_buff, partition_size);
      std::copy(row_numbers.begin(), row_numbers.end(), output_for_partition_buff);
      break;
    }
    case SqlWindowFunctionKind::RANK: {
      const auto rank =
          index_to_rank(output_for_partition_buff, partition_size, col_tuple_comparator);
      std::copy(rank.begin(), rank.end(), output_for_partition_buff);
      break;
    }
    case SqlWindowFunctionKind::DENSE_RANK: {
      const auto dense_rank = index_to_dense_rank(
          output_for_partition_buff, partition_size, col_tuple_comparator);
      std::copy(dense_rank.begin(), dense_rank.end(), output_for_partition_buff);
      break;
    }
    case SqlWindowFunctionKind::PERCENT_RANK: {
      const auto percent_rank = index_to_percent_rank(
          output_for_partition_buff, partition_size, col_tuple_comparator);
      std::copy(percent_rank.begin(),
                percent_rank.end(),
                reinterpret_cast<double*>(may_alias_ptr(output_for_partition_buff)));
      break;
    }
    case SqlWindowFunctionKind::CUME_DIST: {
      const auto cume_dist = index_to_cume_dist(
          output_for_partition_buff, partition_size, col_tuple_comparator);
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
      const auto partition_row_offsets = payload() + offset;
      apply_lag_to_partition(
          lag_or_lead, partition_row_offsets, output_for_partition_buff, partition_size);
      break;
    }
    case SqlWindowFunctionKind::FIRST_VALUE: {
      const auto partition_row_offsets = payload() + offset;
      apply_first_value_to_partition(
          partition_row_offsets, output_for_partition_buff, partition_size);
      break;
    }
    case SqlWindowFunctionKind::LAST_VALUE: {
      const auto partition_row_offsets = payload() + offset;
      apply_last_value_to_partition(
          partition_row_offsets, output_for_partition_buff, partition_size);
      break;
    }
    case SqlWindowFunctionKind::AVG:
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX:
    case SqlWindowFunctionKind::SUM:
    case SqlWindowFunctionKind::COUNT: {
      const auto partition_row_offsets = payload() + offset;
      if (window_function_requires_peer_handling(window_func)) {
        index_to_partition_end(partitionEnd(),
                               offset,
                               output_for_partition_buff,
                               partition_size,
                               col_tuple_comparator);
      }
      apply_permutation_to_partition(
          output_for_partition_buff, partition_row_offsets, partition_size);
      break;
    }
    default: {
      throw std::runtime_error("Window function not supported yet: " +
                               ::toString(window_func->getKind()));
    }
  }
}

void WindowFunctionContext::buildAggregationTreeForPartition(
    SqlWindowFunctionKind agg_type,
    size_t partition_idx,
    size_t partition_size,
    const int8_t* col_buf,
    const int32_t* original_rowid_buf,
    const int64_t* ordered_rowid_buf,
    SQLTypeInfo input_col_ti) {
  const auto type = input_col_ti.is_decimal() ? decimal_to_int_type(input_col_ti)
                                              : input_col_ti.get_type();
  VLOG(1) << "Build Aggregation Tree For Partition-" << ::toString(partition_idx)
          << " (# elems: " << ::toString(partition_size) << ")";
  IndexPair order_col_null_range{-1, -1};
  const int64_t* ordered_rowid_buf_for_partition =
      ordered_rowid_buf + offsets()[partition_idx];
  if (partition_size > 0) {
    // compute null range first
    order_col_null_range = computeNullRangeOfSortedPartition(
        window_func_->getOrderKeys().front()->get_type_info(),
        partition_idx,
        ordered_rowid_buf_for_partition);
    aggregate_tree_null_start_pos_[partition_idx] = order_col_null_range.first;
    aggregate_tree_null_end_pos_[partition_idx] = order_col_null_range.second + 1;
  }
  switch (type) {
    case kTINYINT: {
      std::shared_ptr<SegmentTree<int8_t, int64_t>> segment_tree{nullptr};
      if (partition_size > 0) {
        segment_tree = std::make_shared<SegmentTree<int8_t, int64_t>>(
            col_buf,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            order_col_null_range,
            partition_size,
            agg_type,
            aggregate_trees_fan_out_);
      }
      aggregate_trees_depth_[partition_idx] =
          segment_tree ? segment_tree->getLeafDepth() : 0;
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getDerivedAggregatedValues() : nullptr);
      } else {
        aggregate_trees_.aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getAggregatedValues() : nullptr);
      }
      segment_trees_owned_.emplace_back(std::move(segment_tree));
      break;
    }
    case kSMALLINT: {
      std::shared_ptr<SegmentTree<int16_t, int64_t>> segment_tree{nullptr};
      if (partition_size > 0) {
        segment_tree = std::make_shared<SegmentTree<int16_t, int64_t>>(
            col_buf,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            order_col_null_range,
            partition_size,
            agg_type,
            aggregate_trees_fan_out_);
      }
      aggregate_trees_depth_[partition_idx] =
          segment_tree ? segment_tree->getLeafDepth() : 0;
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getDerivedAggregatedValues() : nullptr);
      } else {
        aggregate_trees_.aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getAggregatedValues() : nullptr);
      }
      segment_trees_owned_.emplace_back(std::move(segment_tree));
      break;
    }
    case kINT: {
      std::shared_ptr<SegmentTree<int32_t, int64_t>> segment_tree{nullptr};
      if (partition_size > 0) {
        segment_tree = std::make_shared<SegmentTree<int32_t, int64_t>>(
            col_buf,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            order_col_null_range,
            partition_size,
            agg_type,
            aggregate_trees_fan_out_);
      }
      aggregate_trees_depth_[partition_idx] =
          segment_tree ? segment_tree->getLeafDepth() : 0;
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getDerivedAggregatedValues() : nullptr);
      } else {
        aggregate_trees_.aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getAggregatedValues() : nullptr);
      }
      segment_trees_owned_.emplace_back(std::move(segment_tree));
      break;
    }
    case kDECIMAL:
    case kNUMERIC:
    case kBIGINT: {
      std::shared_ptr<SegmentTree<int64_t, int64_t>> segment_tree{nullptr};
      if (partition_size > 0) {
        segment_tree = std::make_shared<SegmentTree<int64_t, int64_t>>(
            col_buf,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            order_col_null_range,
            partition_size,
            agg_type,
            aggregate_trees_fan_out_);
      }
      aggregate_trees_depth_[partition_idx] =
          segment_tree ? segment_tree->getLeafDepth() : 0;
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getDerivedAggregatedValues() : nullptr);
      } else {
        aggregate_trees_.aggregate_tree_for_integer_type_.push_back(
            segment_tree ? segment_tree->getAggregatedValues() : nullptr);
      }
      segment_trees_owned_.emplace_back(std::move(segment_tree));
      break;
    }
    case kFLOAT: {
      std::shared_ptr<SegmentTree<float, double>> segment_tree{nullptr};
      if (partition_size > 0) {
        segment_tree =
            std::make_shared<SegmentTree<float, double>>(col_buf,
                                                         input_col_ti,
                                                         original_rowid_buf,
                                                         ordered_rowid_buf_for_partition,
                                                         order_col_null_range,
                                                         partition_size,
                                                         agg_type,
                                                         aggregate_trees_fan_out_);
      }
      aggregate_trees_depth_[partition_idx] =
          segment_tree ? segment_tree->getLeafDepth() : 0;
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_double_type_.push_back(
            segment_tree ? segment_tree->getDerivedAggregatedValues() : nullptr);
      } else {
        aggregate_trees_.aggregate_tree_for_double_type_.push_back(
            segment_tree ? segment_tree->getAggregatedValues() : nullptr);
      }
      segment_trees_owned_.emplace_back(std::move(segment_tree));
      break;
    }
    case kDOUBLE: {
      std::shared_ptr<SegmentTree<double, double>> segment_tree{nullptr};
      if (partition_size > 0) {
        segment_tree =
            std::make_shared<SegmentTree<double, double>>(col_buf,
                                                          input_col_ti,
                                                          original_rowid_buf,
                                                          ordered_rowid_buf_for_partition,
                                                          order_col_null_range,
                                                          partition_size,
                                                          agg_type,
                                                          aggregate_trees_fan_out_);
      }
      aggregate_trees_depth_[partition_idx] =
          segment_tree ? segment_tree->getLeafDepth() : 0;
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_double_type_.push_back(
            segment_tree ? segment_tree->getDerivedAggregatedValues() : nullptr);
      } else {
        aggregate_trees_.aggregate_tree_for_double_type_.push_back(
            segment_tree ? segment_tree->getAggregatedValues() : nullptr);
      }
      segment_trees_owned_.emplace_back(std::move(segment_tree));
      break;
    }
    case kBOOLEAN:
    case kCHAR:
    case kTEXT:
    case kVARCHAR:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kARRAY:
      throw QueryNotSupported("Window aggregate function over frame on a column type " +
                              ::toString(input_col_ti.get_type()) + " is not supported.");
      break;
    default:
      abort();
  }
}

int64_t** WindowFunctionContext::getAggregationTreesForIntegerTypeWindowExpr() const {
  return const_cast<int64_t**>(aggregate_trees_.aggregate_tree_for_integer_type_.data());
}

double** WindowFunctionContext::getAggregationTreesForDoubleTypeWindowExpr() const {
  return const_cast<double**>(aggregate_trees_.aggregate_tree_for_double_type_.data());
}

SumAndCountPair<int64_t>**
WindowFunctionContext::getDerivedAggregationTreesForIntegerTypeWindowExpr() const {
  return const_cast<SumAndCountPair<int64_t>**>(
      aggregate_trees_.derived_aggregate_tree_for_integer_type_.data());
}

SumAndCountPair<double>**
WindowFunctionContext::getDerivedAggregationTreesForDoubleTypeWindowExpr() const {
  return const_cast<SumAndCountPair<double>**>(
      aggregate_trees_.derived_aggregate_tree_for_double_type_.data());
}

size_t* WindowFunctionContext::getAggregateTreeDepth() const {
  return aggregate_trees_depth_;
}

size_t WindowFunctionContext::getAggregateTreeFanout() const {
  return aggregate_trees_fan_out_;
}

int64_t* WindowFunctionContext::getNullValueStartPos() const {
  return aggregate_tree_null_start_pos_;
}

int64_t* WindowFunctionContext::getNullValueEndPos() const {
  return aggregate_tree_null_end_pos_;
}

void WindowFunctionContext::fillPartitionStart() {
  CountDistinctDescriptor partition_start_bitmap{CountDistinctImplType::Bitmap,
                                                 0,
                                                 static_cast<int64_t>(elem_count_),
                                                 false,
                                                 ExecutorDeviceType::CPU,
                                                 1};
  auto bitmap_sz = partition_start_bitmap.bitmapPaddedSizeBytes();
  if (partitions_) {
    bitmap_sz += partitions_->isBitwiseEq() ? 1 : 0;
  }
  partition_start_ = static_cast<int8_t*>(checked_calloc(bitmap_sz, 1));
  int64_t partition_count = partitionCount();
  auto partition_start_handle = reinterpret_cast<int64_t>(partition_start_);
  agg_count_distinct_bitmap(&partition_start_handle, 0, 0);
  if (partition_start_offset_) {
    // if we have `partition_start_offset_`, we can reuse it for this logic
    // but note that it has partition_count + 1 elements where the first element is zero
    // which means the first partition's start offset is zero
    // and rest of them can represent values required for this logic
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      agg_count_distinct_bitmap(
          &partition_start_handle, partition_start_offset_[i + 1], 0);
    }
  } else {
    std::vector<size_t> partition_offsets(partition_count);
    std::partial_sum(counts(), counts() + partition_count, partition_offsets.begin());
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      agg_count_distinct_bitmap(&partition_start_handle, partition_offsets[i], 0);
    }
  }
}

void WindowFunctionContext::fillPartitionEnd() {
  CountDistinctDescriptor partition_start_bitmap{CountDistinctImplType::Bitmap,
                                                 0,
                                                 static_cast<int64_t>(elem_count_),
                                                 false,
                                                 ExecutorDeviceType::CPU,
                                                 1};
  auto bitmap_sz = partition_start_bitmap.bitmapPaddedSizeBytes();
  if (partitions_) {
    bitmap_sz += partitions_->isBitwiseEq() ? 1 : 0;
  }
  partition_end_ = static_cast<int8_t*>(checked_calloc(bitmap_sz, 1));
  auto partition_end_handle = reinterpret_cast<int64_t>(partition_end_);
  int64_t partition_count = partitionCount();
  if (partition_start_offset_) {
    // if we have `partition_start_offset_`, we can reuse it for this logic
    // but note that it has partition_count + 1 elements where the first element is zero
    // which means the first partition's start offset is zero
    // and rest of them can represent values required for this logic
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      if (partition_start_offset_[i + 1] == 0) {
        continue;
      }
      agg_count_distinct_bitmap(
          &partition_end_handle, partition_start_offset_[i + 1] - 1, 0);
    }
    if (elem_count_) {
      agg_count_distinct_bitmap(&partition_end_handle, elem_count_ - 1, 0);
    }
  } else {
    std::vector<size_t> partition_offsets(partition_count);
    std::partial_sum(counts(), counts() + partition_count, partition_offsets.begin());
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      if (partition_offsets[i] == 0) {
        continue;
      }
      agg_count_distinct_bitmap(&partition_end_handle, partition_offsets[i] - 1, 0);
    }
    if (elem_count_) {
      agg_count_distinct_bitmap(&partition_end_handle, elem_count_ - 1, 0);
    }
  }
}

const int32_t* WindowFunctionContext::payload() const {
  if (partitions_) {
    return reinterpret_cast<const int32_t*>(
        partitions_->getJoinHashBuffer(device_type_, 0) +
        partitions_->payloadBufferOff());
  }
  return dummy_payload_;  // non-partitioned window function
}

const int32_t* WindowFunctionContext::offsets() const {
  if (partitions_) {
    return reinterpret_cast<const int32_t*>(
        partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->offsetBufferOff());
  }
  return &dummy_offset_;
}

const int32_t* WindowFunctionContext::counts() const {
  if (partitions_) {
    return reinterpret_cast<const int32_t*>(
        partitions_->getJoinHashBuffer(device_type_, 0) + partitions_->countBufferOff());
  }
  return &dummy_count_;
}

size_t WindowFunctionContext::partitionCount() const {
  if (partitions_) {
    const auto partition_count = counts() - offsets();
    CHECK_GE(partition_count, 0);
    return partition_count;
  }
  return 1;  // non-partitioned window function
}

void WindowProjectNodeContext::addWindowFunctionContext(
    std::unique_ptr<WindowFunctionContext> window_function_context,
    const size_t target_index) {
  const auto it_ok = window_contexts_.emplace(
      std::make_pair(target_index, std::move(window_function_context)));
  CHECK(it_ok.second);
}

const WindowFunctionContext* WindowProjectNodeContext::activateWindowFunctionContext(
    Executor* executor,
    const size_t target_index) const {
  const auto it = window_contexts_.find(target_index);
  CHECK(it != window_contexts_.end());
  executor->active_window_function_ = it->second.get();
  return executor->active_window_function_;
}

void WindowProjectNodeContext::resetWindowFunctionContext(Executor* executor) {
  executor->active_window_function_ = nullptr;
}

WindowFunctionContext* WindowProjectNodeContext::getActiveWindowFunctionContext(
    Executor* executor) {
  return executor->active_window_function_;
}

WindowProjectNodeContext* WindowProjectNodeContext::create(Executor* executor) {
  executor->window_project_node_context_owned_ =
      std::make_unique<WindowProjectNodeContext>();
  return executor->window_project_node_context_owned_.get();
}

const WindowProjectNodeContext* WindowProjectNodeContext::get(Executor* executor) {
  return executor->window_project_node_context_owned_.get();
}

void WindowProjectNodeContext::reset(Executor* executor) {
  executor->window_project_node_context_owned_ = nullptr;
  executor->active_window_function_ = nullptr;
}
