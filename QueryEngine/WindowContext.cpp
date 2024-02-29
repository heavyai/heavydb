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
#include "Shared/scope.h"
#include "Shared/sqltypes.h"
#include "Shared/threading.h"

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#endif

#ifdef HAVE_CUDA
CUstream getQueryEngineCudaStreamForDevice(int device_num);
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
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    Data_Namespace::DataMgr* data_mgr,
    DeviceAllocator* device_allocator,
    const int device_id,
    size_t aggregation_tree_fan_out)
    : window_func_(window_func)
    , partition_cache_key_(EMPTY_HASHED_PLAN_DAG_KEY)
    , sorted_partition_cache_key_(EMPTY_HASHED_PLAN_DAG_KEY)
    , elem_count_(elem_count)
    , partitions_(nullptr)
    , row_set_mem_owner_(row_set_mem_owner)
    , data_mgr_(data_mgr)
    , device_type_(device_type)
    , device_id_(device_id)
    , device_allocator_(device_allocator)
    , aggregate_trees_fan_out_(aggregation_tree_fan_out) {
  CHECK_LE(elem_count_, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  auto const for_gpu = device_type_ == ExecutorDeviceType::GPU;
  if (for_gpu) {
    gpu_exec_ctx_.data_mgr_ = data_mgr_;
  }
  // a column becomes a single and whole partition
  size_t partition_count = 1;
  prepareEmptyPartition(for_gpu);
  if (window_func_->hasFraming() || window_func_->isMissingValueFillingFunction() ||
      window_func_->getKind() == SqlWindowFunctionKind::NTH_VALUE) {
    aggregate_trees_.data_mgr_ = data_mgr_;
    preparePartitionStartOffsetBufs(for_gpu, partition_count);
    prepareAggregateTreeBufs(for_gpu, partition_count);
  }
}

// Partitioned version
WindowFunctionContext::WindowFunctionContext(
    const Analyzer::WindowFunction* window_func,
    QueryPlanHash cache_key,
    std::shared_ptr<HashJoin> hash_table,
    const size_t elem_count,
    const ExecutorDeviceType device_type,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    Data_Namespace::DataMgr* data_mgr,
    DeviceAllocator* device_allocator,
    const int device_id,
    size_t aggregation_tree_fan_out)
    : window_func_(window_func)
    , partition_cache_key_(cache_key)
    , sorted_partition_cache_key_(EMPTY_HASHED_PLAN_DAG_KEY)
    , elem_count_(elem_count)
    , partitions_(hash_table)
    , row_set_mem_owner_(row_set_mem_owner)
    , data_mgr_(data_mgr)
    , device_type_(device_type)
    , device_id_(device_id)
    , device_allocator_(device_allocator)
    , aggregate_trees_fan_out_(aggregation_tree_fan_out) {
  CHECK(partitions_);  // This version should have hash table
  auto const for_gpu = device_type_ == ExecutorDeviceType::GPU;
  if (for_gpu) {
    gpu_exec_ctx_.data_mgr_ = data_mgr_;
  }
  size_t partition_count = getNumWindowPartition();
  if (window_func_->hasFraming() || window_func_->isMissingValueFillingFunction() ||
      window_func_->getKind() == SqlWindowFunctionKind::NTH_VALUE) {
    aggregate_trees_.data_mgr_ = data_mgr_;
    preparePartitionStartOffsetBufs(for_gpu, partition_count);
    prepareAggregateTreeBufs(for_gpu, partition_count);
  }
}

void WindowFunctionContext::addOrderColumn(
    const int8_t* column,
    const SQLTypeInfo& ti,
    const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::CPU) {
    column_chunk_owned_.order_columns_owner_.push_back(chunks_owner);
    column_chunk_owned_.order_columns_.push_back(column);
  } else {
    column_chunk_owned_.order_columns_for_gpu_owner_.push_back(chunks_owner);
    column_chunk_owned_.order_columns_for_gpu_.push_back(column);
  }
  column_chunk_owned_.order_columns_ti_.push_back(ti);
}

void WindowFunctionContext::addColumnBufferForWindowFunctionExpression(
    const int8_t* column,
    const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::CPU) {
    column_chunk_owned_.window_func_expr_columns_owner_.push_back(chunks_owner);
    column_chunk_owned_.window_func_expr_columns_.push_back(column);
  } else {
    column_chunk_owned_.window_func_expr_columns_for_gpu_owner_.push_back(chunks_owner);
    column_chunk_owned_.window_func_expr_columns_for_gpu_.push_back(column);
  }
};

const std::vector<const int8_t*>&
WindowFunctionContext::getColumnBufferForWindowFunctionExpressions() const {
  return device_type_ == ExecutorDeviceType::CPU
             ? column_chunk_owned_.window_func_expr_columns_
             : column_chunk_owned_.window_func_expr_columns_for_gpu_;
}

const std::vector<const int8_t*>& WindowFunctionContext::getOrderKeyColumnBuffers()
    const {
  return device_type_ == ExecutorDeviceType::CPU
             ? column_chunk_owned_.order_columns_
             : column_chunk_owned_.order_columns_for_gpu_;
}

const std::vector<SQLTypeInfo>& WindowFunctionContext::getOrderKeyColumnBufferTypes()
    const {
  return column_chunk_owned_.order_columns_ti_;
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

size_t get_target_idx_for_first_or_last_value_func(
    const Analyzer::WindowFunction* window_func,
    const size_t partition_size) {
  CHECK(window_func->getKind() == SqlWindowFunctionKind::FIRST_VALUE ||
        window_func->getKind() == SqlWindowFunctionKind::LAST_VALUE);
  return window_func->getKind() == SqlWindowFunctionKind::FIRST_VALUE
             ? 0
             : partition_size - 1;
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

void apply_nth_value_to_partition(const int32_t* original_indices,
                                  int64_t* output_for_partition_buff,
                                  const size_t partition_size,
                                  const size_t target_pos) {
  CHECK_LT(target_pos, partition_size);
  const auto target_idx = original_indices[output_for_partition_buff[target_pos]];
  std::fill(
      output_for_partition_buff, output_for_partition_buff + partition_size, target_idx);
}

void apply_original_index_to_partition(const int32_t* original_indices,
                                       int64_t* output_for_partition_buff,
                                       const size_t partition_size) {
  for (size_t i = 0; i < partition_size; i++) {
    const auto target_idx = original_indices[output_for_partition_buff[i]];
    output_for_partition_buff[i] = target_idx;
  }
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
      agg_count_distinct_bitmap(&partition_end_handle, off + i - 1, 0, 0);
    }
  }
  CHECK(index_size);
  agg_count_distinct_bitmap(&partition_end_handle, off + index_size - 1, 0, 0);
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
    std::unordered_map<QueryPlanHash, int8_t*>& sorted_partition_cache,
    std::unordered_map<size_t, WindowFunctionCtx::AggregateTreeForWindowFraming>&
        aggregate_tree_map) {
  // 1. init and precondition checking
  auto compute_timer = DEBUG_TIMER(__func__);
  CHECK(!cpu_exec_ctx_.output_);
  if (elem_count_ == 0) {
    return;
  }
  // 2. output buffer allocation
  size_t output_buf_sz =
      elem_count_ * window_function_buffer_element_size(window_func_->getKind());
  cpu_exec_ctx_.output_ =
      static_cast<int8_t*>(row_set_mem_owner_->allocate(output_buf_sz,
                                                        /*thread_idx=*/0));
  // 3. determine window function's characteristics
  bool const is_agg_func = window_function_is_aggregate(window_func_->getKind());
  bool const use_aggregation_tree =
      window_func_->hasFraming() || window_func_->isMissingValueFillingFunction();

  // 4. create two bitmaps to represent each window partition's start and end offsets
  if (is_agg_func || use_aggregation_tree) {
    fillPartitionStart();
    if (window_function_requires_peer_handling(window_func_) || use_aggregation_tree) {
      fillPartitionEnd();
    }
  }

  // 5. initialize intermediate output buffer to compute window function
  std::unique_ptr<int64_t[]> intermediate_output_buffer_;
  if (is_agg_func || use_aggregation_tree) {
    cpu_exec_ctx_.intermediate_output_buffer_ =
        reinterpret_cast<int64_t*>(cpu_exec_ctx_.output_);
  } else {
    output_buf_sz = sizeof(int64_t) * elem_count_;
    intermediate_output_buffer_ = std::make_unique<int64_t[]>(elem_count_);
    cpu_exec_ctx_.intermediate_output_buffer_ = intermediate_output_buffer_.get();
  }
  ScopeGuard invalidate_intermediate_output_buffer = [this] {
    cpu_exec_ctx_.intermediate_output_buffer_ = nullptr;
  };

  // 6. sorting and caching window partitions if necessary
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
    auto partition_copy_timer =
        DEBUG_TIMER("Window Function Cached Sorted Partition Copy");
    std::memcpy(
        cpu_exec_ctx_.intermediate_output_buffer_, sorted_partition, output_buf_sz);
  } else {
    // to sort window partition, we first fill the `intermediate_output_buffer_`
    // by calling std::iota (0 ~ N) where N is the # elems
    // so each elem indicates the current row's index within a window partition
    // now, if we have ordering column(s), we sort the window partition based on it/them
    // the result of ordering changes the values stored in the
    // `intermediate_output_buffer_` and it now represents a permutation index of the
    // original ordering column here, values stored in the window partition for a
    // specific partition-i (i.e., payload() + offsets()[i]) represent an actual row ids
    // (0 ~ N) but in the corresponding permutation index for the same partition-i, it
    // has 0 ~ counts()[i] where counts()[i] is the # elements in the partition-i
    const auto sort_partitions = [&](const size_t start, const size_t end) {
      for (size_t partition_idx = start; partition_idx < end; ++partition_idx) {
        sortPartition(partition_idx,
                      cpu_exec_ctx_.intermediate_output_buffer_ +
                          getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx]);
      }
    };

    if (should_parallelize) {
      auto sorted_partition_copy_timer =
          DEBUG_TIMER("Window Function Partition Sorting Parallelized");
#if HAVE_TBB
      tbb::parallel_for(tbb::blocked_range<int64_t>(0, getNumWindowPartition()),
                        [&, parent_thread_local_ids = logger::thread_local_ids()](
                            const tbb::blocked_range<int64_t>& r) {
                          logger::LocalIdsScopeGuard lisg =
                              parent_thread_local_ids.setNewThreadId();
                          sort_partitions(r.begin(), r.end());
                        });
#else
      std::vector<std::future<void>> workers;
      auto const worker_count = cpu_threads();
      for (auto interval :
           makeIntervals(size_t(0), getNumWindowPartition(), worker_count)) {
        workers.push_back(std::async(
            std::launch::async, sort_partitions, interval.begin, interval.end));
      }
      for (auto& worker : workers) {
        worker.get();
      }
#endif
    } else {
      auto sorted_partition_copy_timer =
          DEBUG_TIMER("Window Function  Partition Sorting Non-Parallelized");
      sort_partitions(0, getNumWindowPartition());
    }
    auto sorted_partition_ref_cnt_it =
        sorted_partition_key_ref_count_map.find(sorted_partition_cache_key_);
    bool keep_sorted_partition_in_cache =
        sorted_partition_ref_cnt_it != sorted_partition_key_ref_count_map.end() &&
        sorted_partition_ref_cnt_it->second > 1 &&
        sorted_partition_cache.find(sorted_partition_cache_key_) ==
            sorted_partition_cache.end();
    if (keep_sorted_partition_in_cache) {
      // keep the sorted partition only if it will be reused from other window function
      // context of this query
      auto sorted_partition_memcpy_timer =
          DEBUG_TIMER("Window Function Sorted Partition Caching");
      auto copied_sorted_partition_buf =
          static_cast<int8_t*>(row_set_mem_owner_->allocate(output_buf_sz, 0));
      std::memcpy(copied_sorted_partition_buf,
                  cpu_exec_ctx_.intermediate_output_buffer_,
                  output_buf_sz);
      auto it = sorted_partition_cache.emplace(sorted_partition_cache_key_,
                                               copied_sorted_partition_buf);
      if (it.second) {
        VLOG(1) << "Put sorted partition to cache (key: " << sorted_partition_cache_key_
                << ", ordering condition: " << ::toString(window_func_->getOrderKeys())
                << ")";
      }
    }
  }
  {
    auto sorted_partition_memcpy_timer =
        DEBUG_TIMER("Window Function Sorted Partition Copy");
    cpu_exec_ctx_.sorted_partition_buf_ =
        static_cast<int8_t*>(row_set_mem_owner_->allocate(output_buf_sz, 0));
    std::memcpy(cpu_exec_ctx_.sorted_partition_buf_,
                cpu_exec_ctx_.intermediate_output_buffer_,
                output_buf_sz);
    if (device_type_ == ExecutorDeviceType::GPU) {
      auto sorted_partition_d2h_timer =
          DEBUG_TIMER("Window Function Sorted Partition Copy To Device");
      gpu_exec_ctx_.sorted_partition_buf_gpu_ =
          data_mgr_->alloc(MemoryLevel::GPU_LEVEL, device_id_, output_buf_sz);
      device_allocator_->copyToDevice(
          gpu_exec_ctx_.sorted_partition_buf_gpu_->getMemoryPtr(),
          cpu_exec_ctx_.sorted_partition_buf_,
          output_buf_sz,
          "WindowFunc_SortedPartition");
    }
  }

  // 7. build an aggregate tree if we need to compute this window function based on
  // framing
  if (use_aggregation_tree) {
    const auto compute_ordered_partition_null_range = [=](const size_t start,
                                                          const size_t end) {
      for (size_t partition_idx = start; partition_idx < end; ++partition_idx) {
        computeNullRangeOfSortedPartition(
            window_func_->getOrderKeys().front()->get_type_info(),
            partition_idx,
            getPayloadBuf(ExecutorDeviceType::CPU) +
                getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx],
            cpu_exec_ctx_.intermediate_output_buffer_ +
                getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx]);
      }
    };
    auto partition_count = getNumWindowPartition();

    if (should_parallelize) {
      auto partition_computation_timer =
          DEBUG_TIMER("Window Function Ordered-Partition Null-Range Compute");
#if HAVE_TBB
      tbb::parallel_for(tbb::blocked_range<int64_t>(0, getNumWindowPartition()),
                        [&, parent_thread_local_ids = logger::thread_local_ids()](
                            const tbb::blocked_range<int64_t>& r) {
                          logger::LocalIdsScopeGuard lisg =
                              parent_thread_local_ids.setNewThreadId();
                          compute_ordered_partition_null_range(r.begin(), r.end());
                        });
#else
      std::vector<std::future<void>> workers;
      auto const worker_count = cpu_threads();
      for (auto interval :
           makeIntervals(size_t(0), getNumWindowPartition(), worker_count)) {
        workers.push_back(std::async(std::launch::async,
                                     compute_ordered_partition_null_range,
                                     interval.begin,
                                     interval.end));
      }
      for (auto& worker : workers) {
        worker.get();
      }
#endif
    } else {
      auto partition_computation_timer = DEBUG_TIMER(
          "Window Function Non-Parallelized Ordered-Partition Null-Range Compute");
      compute_ordered_partition_null_range(0, getNumWindowPartition());
    }

    auto const cache_key = computeAggregateTreeCacheKey();
    size_t total_buf_size = 0;
    auto const agg_type = window_func_->getKind();
    auto input_col_ti = window_func_->getArgs().front()->get_type_info();
    auto const type = input_col_ti.is_decimal() ? decimal_to_int_type(input_col_ti)
                      : input_col_ti.is_time_or_date()
                          ? get_int_type_by_size(input_col_ti.get_size())
                          : input_col_ti.get_type();
    auto const c_it = aggregate_tree_map.find(cache_key);
    if (needsToBuildAggregateTree()) {
      if (c_it != aggregate_tree_map.cend()) {
        VLOG(1) << "Reuse aggregate tree for window function framing";
        resizeStorageForWindowFraming(true);
        aggregate_trees_ = c_it->second;
        std::for_each(aggregate_trees_.aggregate_tree_sizes_in_byte_.begin(),
                      aggregate_trees_.aggregate_tree_sizes_in_byte_.end(),
                      [&total_buf_size](size_t size) { total_buf_size += size; });
      } else {
        // Allocate a contiguous memory space that holds all aggregate trees for the
        // window framing First, calculate 1) the total size of memory we need to allocate
        // and 2)
        resizeStorageForWindowFraming();
        // compute each aggregate tree's size
        for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
          auto const partition_size = getCountBuf(ExecutorDeviceType::CPU)[partition_idx];
          auto const tree_info =
              compute_segment_tree_info(partition_size, aggregate_trees_fan_out_);
          aggregate_trees_.aggregate_tree_infos_[partition_idx] = tree_info;
          size_t num_nodes_in_tree = tree_info.second.second;
          aggregate_trees_.aggregate_tree_sizes_[partition_idx] = num_nodes_in_tree;
          if (partition_size > 0) {
            size_t tree_size = 0;
            switch (type) {
              case kBOOLEAN:
              case kTINYINT:
              case kSMALLINT:
              case kINT:
              case kDECIMAL:
              case kNUMERIC:
              case kBIGINT:
                if (agg_type == SqlWindowFunctionKind::AVG) {
                  tree_size = num_nodes_in_tree * sizeof(SumAndCountPair<int64_t>);
                } else {
                  tree_size = num_nodes_in_tree * sizeof(int64_t);
                }
                break;
              case kFLOAT:
              case kDOUBLE:
                if (agg_type == SqlWindowFunctionKind::AVG) {
                  tree_size = num_nodes_in_tree * sizeof(SumAndCountPair<double>);
                } else {
                  tree_size = num_nodes_in_tree * sizeof(double);
                }
                break;
              default:
                UNREACHABLE() << type;
            }
            aggregate_trees_.aggregate_tree_sizes_in_byte_[partition_idx] = tree_size;
            total_buf_size += tree_size;
          }
        }
        CHECK_EQ(aggregate_trees_.aggregate_tree_infos_.size(),
                 static_cast<size_t>(partition_count));
        // Second, compute start offsets for each chunk which is corresponding to an
        // aggregate tree
        std::exclusive_scan(aggregate_trees_.aggregate_tree_sizes_in_byte_.begin(),
                            aggregate_trees_.aggregate_tree_sizes_in_byte_.end(),
                            aggregate_trees_.aggregate_tree_start_offset_.begin(),
                            static_cast<size_t>(0));
        // Third, divide the contiguous memory into N chunks where N is equal to the
        // partition count
        VLOG(2) << "Allocate " << total_buf_size
                << " bytes on CPU memory to build aggregate tree(s)";
        aggregate_trees_.aggregate_tree_holder_ =
            row_set_mem_owner_->allocate(total_buf_size, 0);
        for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
          auto const start_offset =
              aggregate_trees_.aggregate_tree_start_offset_[partition_idx];
          switch (type) {
            case kBOOLEAN:
            case kTINYINT:
            case kSMALLINT:
            case kINT:
            case kDECIMAL:
            case kNUMERIC:
            case kBIGINT:
              if (agg_type == SqlWindowFunctionKind::AVG) {
                aggregate_trees_.derived_aggregate_tree_for_integer_type_[partition_idx] =
                    reinterpret_cast<SumAndCountPair<int64_t>*>(
                        aggregate_trees_.aggregate_tree_holder_ + start_offset);

              } else {
                aggregate_trees_.aggregate_tree_for_integer_type_[partition_idx] =
                    reinterpret_cast<int64_t*>(aggregate_trees_.aggregate_tree_holder_ +
                                               start_offset);
              }
              break;
            case kFLOAT:
            case kDOUBLE:
              if (agg_type == SqlWindowFunctionKind::AVG) {
                aggregate_trees_.derived_aggregate_tree_for_double_type_[partition_idx] =
                    reinterpret_cast<SumAndCountPair<double>*>(
                        aggregate_trees_.aggregate_tree_holder_ + start_offset);
              } else {
                aggregate_trees_.aggregate_tree_for_double_type_[partition_idx] =
                    reinterpret_cast<double*>(aggregate_trees_.aggregate_tree_holder_ +
                                              start_offset);
              }
              break;
            default:
              UNREACHABLE() << type;
          }
        }
        const auto build_aggregation_tree_for_partitions = [=](const size_t start,
                                                               const size_t end) {
          for (size_t partition_idx = start; partition_idx < end; ++partition_idx) {
            // build a segment tree for the partition
            // todo (yoonmin) : support generic window function expression
            // i.e., when window_func_expr_columns_.size() > 1
            SQLTypeInfo const input_col_ti =
                window_func_->getArgs().front()->get_type_info();
            const auto partition_size =
                getCountBuf(ExecutorDeviceType::CPU)[partition_idx];
            buildAggregationTreeForPartition(
                agg_type,
                partition_idx,
                partition_size,
                getPayloadBuf(ExecutorDeviceType::CPU) +
                    getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx],
                cpu_exec_ctx_.intermediate_output_buffer_,
                input_col_ti);
          }
        };
        if (should_parallelize) {
          auto aggregate_tree_builder_timer = DEBUG_TIMER(
              "Window Function Parallelized Segment Tree Construction for Partitions");
#if HAVE_TBB
          tbb::parallel_for(tbb::blocked_range<int64_t>(0, getNumWindowPartition()),
                            [=, parent_thread_local_ids = logger::thread_local_ids()](
                                const tbb::blocked_range<int64_t>& r) {
                              logger::LocalIdsScopeGuard lisg =
                                  parent_thread_local_ids.setNewThreadId();
                              build_aggregation_tree_for_partitions(r.begin(), r.end());
                            });
#else
          std::vector<std::future<void>> workers;
          auto const worker_count = cpu_threads();
          for (auto interval :
               makeIntervals(size_t(0), getNumWindowPartition(), worker_count)) {
            workers.push_back(std::async(std::launch::async,
                                         build_aggregation_tree_for_partitions,
                                         interval.begin,
                                         interval.end));
          }
          for (auto& worker : workers) {
            worker.get();
          }
#endif
        } else {
          auto aggregate_tree_builder_timer = DEBUG_TIMER(
              "Window Function Non-Parallelized Segment Tree Construction for "
              "Partitions");
          build_aggregation_tree_for_partitions(0, getNumWindowPartition());
        }
        CHECK(aggregate_tree_map.emplace(cache_key, aggregate_trees_).second);
        VLOG(2) << "Put aggregate tree for the window framing";
      }
    }
    if (device_type_ == ExecutorDeviceType::GPU) {
      auto aggregate_tree_gpu_copy_timer =
          DEBUG_TIMER("Window Function Segment Trees GPU Copy");
      // 1. copy "all" aggregate trees to gpu
      std::vector<void*> dev_buf_ptrs(partition_count);
      aggregate_trees_.aggregate_trees_holder_gpu_ =
          data_mgr_->alloc(MemoryLevel::GPU_LEVEL, device_id_, total_buf_size);
      auto dev_aggregate_tree_ptr =
          aggregate_trees_.aggregate_trees_holder_gpu_->getMemoryPtr();
      device_allocator_->copyToDevice(dev_aggregate_tree_ptr,
                                      aggregate_trees_.aggregate_tree_holder_,
                                      total_buf_size,
                                      "WindowFunc_AggTreePtrsBuf");
      // 2. prepare each aggregate tree's ptr
      for (size_t i = 0; i < partition_count; ++i) {
        auto const tree_start_offset = aggregate_trees_.aggregate_tree_start_offset_[i];
        auto const tree_size = aggregate_trees_.aggregate_tree_sizes_[i];
        if (tree_size > 0) {
          switch (type) {
            case kFLOAT:
            case kDOUBLE: {
              if (window_func_->getKind() == SqlWindowFunctionKind::AVG) {
                dev_buf_ptrs[i] = reinterpret_cast<SumAndCountPair<double>*>(
                    dev_aggregate_tree_ptr + tree_start_offset);
              } else {
                dev_buf_ptrs[i] =
                    reinterpret_cast<double*>(dev_aggregate_tree_ptr + tree_start_offset);
              }
              break;
            }
            default: {
              if (window_func_->getKind() == SqlWindowFunctionKind::AVG) {
                dev_buf_ptrs[i] = reinterpret_cast<SumAndCountPair<int64_t>*>(
                    dev_aggregate_tree_ptr + tree_start_offset);
              } else {
                dev_buf_ptrs[i] = reinterpret_cast<int64_t*>(dev_aggregate_tree_ptr +
                                                             tree_start_offset);
              }
              break;
            }
          }
        } else {
          dev_buf_ptrs[i] = nullptr;
        }
      }
      // 3. copy device memory ptrs to device memory ptr holder
      switch (type) {
        case kFLOAT:
        case kDOUBLE:
          if (window_func_->getKind() == SqlWindowFunctionKind::AVG) {
            device_allocator_->copyToDevice(
                aggregate_trees_.derived_aggregate_tree_for_double_type_gpu_,
                dev_buf_ptrs.data(),
                sizeof(SumAndCountPair<double>*) * partition_count,
                "WindowFunc_DerivedAggTreeForDoubleTy");
          } else {
            device_allocator_->copyToDevice(
                aggregate_trees_.aggregate_tree_for_double_type_gpu_,
                dev_buf_ptrs.data(),
                sizeof(double*) * partition_count,
                "WindowFunc_AggTreeForDoubleTy");
          }
          break;
        default:
          if (window_func_->getKind() == SqlWindowFunctionKind::AVG) {
            device_allocator_->copyToDevice(
                aggregate_trees_.derived_aggregate_tree_for_integer_type_gpu_,
                dev_buf_ptrs.data(),
                sizeof(SumAndCountPair<int64_t*>*) * partition_count,
                "WindowFunc_DerivedAggTreeForIntTy");
          } else {
            device_allocator_->copyToDevice(
                aggregate_trees_.aggregate_tree_for_integer_type_gpu_,
                dev_buf_ptrs.data(),
                sizeof(int64_t*) * partition_count,
                "WindowFunc_AggTreeForIntTy");
          }
          break;
      }
      device_allocator_->copyToDevice(
          aggregate_trees_.aggregate_trees_depth_gpu_->getMemoryPtr(),
          aggregate_trees_.aggregate_trees_depth_.data(),
          partition_count * sizeof(size_t),
          "WindowFunc_AggTreeDepth");
      device_allocator_->copyToDevice(
          aggregate_trees_.aggregate_trees_leaf_start_offset_gpu_->getMemoryPtr(),
          aggregate_trees_.aggregate_trees_leaf_start_offset_.data(),
          partition_count * sizeof(size_t),
          "WindowFunc_AggTreeLeftStartOffset");
    }
  }
  // 8. compute window partition
  const auto compute_partitions = [=](const size_t start, const size_t end) {
    for (size_t partition_idx = start; partition_idx < end; ++partition_idx) {
      computePartitionBuffer(partition_idx,
                             cpu_exec_ctx_.intermediate_output_buffer_ +
                                 getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx],
                             window_func_);
    }
  };

  if (should_parallelize) {
    auto partition_computation_timer = DEBUG_TIMER("Window Function Partition Compute");
#if HAVE_TBB
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, getNumWindowPartition()),
                      [&, parent_thread_local_ids = logger::thread_local_ids()](
                          const tbb::blocked_range<int64_t>& r) {
                        logger::LocalIdsScopeGuard lisg =
                            parent_thread_local_ids.setNewThreadId();
                        compute_partitions(r.begin(), r.end());
                      });
#else
    std::vector<std::future<void>> workers;
    auto const worker_count = cpu_threads();
    for (auto interval :
         makeIntervals(size_t(0), getNumWindowPartition(), worker_count)) {
      workers.push_back(std::async(
          std::launch::async, compute_partitions, interval.begin, interval.end));
    }
    for (auto& worker : workers) {
      worker.get();
    }
#endif
  } else {
    auto partition_computation_timer =
        DEBUG_TIMER("Window Function Non-Parallelized Partition Compute");
    compute_partitions(0, getNumWindowPartition());
  }

  // 9. finalize the computed window partition
  if (is_agg_func || use_aggregation_tree) {
    // If window function is aggregate we were able to write to the final output buffer
    // directly in computePartition and we are done.
    return;
  }

  auto output_i64 = reinterpret_cast<int64_t*>(cpu_exec_ctx_.output_);
  const auto payload_copy = [=](const size_t start, const size_t end) {
    for (size_t i = start; i < end; ++i) {
      output_i64[getPayloadBuf(ExecutorDeviceType::CPU)[i]] =
          cpu_exec_ctx_.intermediate_output_buffer_[i];
    }
  };
  if (should_parallelize) {
    auto payload_copy_timer =
        DEBUG_TIMER("Window Function Non-Aggregate Payload Copy Parallelized");
#if HAVE_TBB
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, elem_count_),
                      [&, parent_thread_local_ids = logger::thread_local_ids()](
                          const tbb::blocked_range<int64_t>& r) {
                        logger::LocalIdsScopeGuard lisg =
                            parent_thread_local_ids.setNewThreadId();
                        payload_copy(r.begin(), r.end());
                      });
#else
    std::vector<std::future<void>> workers;
    auto const worker_count = cpu_threads();
    for (auto interval :
         makeIntervals(size_t(0), getNumWindowPartition(), worker_count)) {
      workers.push_back(
          std::async(std::launch::async, payload_copy, interval.begin, interval.end));
    }
    for (auto& worker : workers) {
      worker.get();
    }
#endif
  } else {
    auto payload_copy_timer =
        DEBUG_TIMER("Window Function Non-Aggregate Payload Copy Non-Parallelized");
    payload_copy(0, elem_count_);
  }
}

namespace {
struct FindNullRange {
  int32_t const* original_col_idx_buf;
  int64_t const* ordered_col_idx_buf;
  int32_t const partition_size;
  int64_t null_bit_pattern = -1;

  template <typename T>
  IndexPair find_null_range_int(int8_t const* order_col_buf) const {
    IndexPair null_range{std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<int64_t>::min()};
    auto const null_val = inline_int_null_value<T>();
    auto const casted_order_col_buf = reinterpret_cast<T const*>(order_col_buf);
    if (casted_order_col_buf[original_col_idx_buf[ordered_col_idx_buf[0]]] == null_val) {
      int64_t null_range_max = 1;
      while (null_range_max < partition_size &&
             casted_order_col_buf
                     [original_col_idx_buf[ordered_col_idx_buf[null_range_max]]] ==
                 null_val) {
        null_range_max++;
      }
      null_range.first = 0;
      null_range.second = null_range_max - 1;
    } else if (casted_order_col_buf
                   [original_col_idx_buf[ordered_col_idx_buf[partition_size - 1]]] ==
               null_val) {
      int64_t null_range_min = partition_size - 2;
      while (null_range_min >= 0 &&
             casted_order_col_buf
                     [original_col_idx_buf[ordered_col_idx_buf[null_range_min]]] ==
                 null_val) {
        null_range_min--;
      }
      null_range.first = null_range_min + 1;
      null_range.second = partition_size - 1;
    }
    return null_range;
  }

  template <typename COL_TYPE,
            typename NULL_TYPE =
                std::conditional_t<sizeof(COL_TYPE) == sizeof(int32_t), int32_t, int64_t>>
  IndexPair find_null_range_fp(int8_t const* order_col_buf) const {
    IndexPair null_range{std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<int64_t>::min()};
    auto const casted_order_col_buf = reinterpret_cast<COL_TYPE const*>(order_col_buf);
    auto check_null_val = [&casted_order_col_buf, this](size_t idx) {
      return *reinterpret_cast<NULL_TYPE const*>(
                 may_alias_ptr(&casted_order_col_buf
                                   [original_col_idx_buf[ordered_col_idx_buf[idx]]])) ==
             null_bit_pattern;
    };
    if (check_null_val(0)) {
      int64_t null_range_max = 1;
      while (null_range_max < partition_size && check_null_val(null_range_max)) {
        null_range_max++;
      }
      null_range.first = 0;
      null_range.second = null_range_max - 1;
    } else if (check_null_val(partition_size - 1)) {
      int64_t null_range_min = partition_size - 2;
      while (null_range_min >= 0 && check_null_val(null_range_min)) {
        null_range_min--;
      }
      null_range.first = null_range_min + 1;
      null_range.second = partition_size - 1;
    }
    return null_range;
  }
};
}  // namespace

void WindowFunctionContext::computeNullRangeOfSortedPartition(
    const SQLTypeInfo& order_col_ti,
    size_t partition_idx,
    const int32_t* original_col_idx_buf,
    const int64_t* ordered_col_idx_buf) {
  IndexPair null_range;
  const auto partition_size = getCountBuf(ExecutorDeviceType::CPU)[partition_idx];
  if (partition_size > 0) {
    if (order_col_ti.is_integer() || order_col_ti.is_decimal() ||
        order_col_ti.is_time_or_date() || order_col_ti.is_boolean()) {
      FindNullRange const null_range_info{
          original_col_idx_buf, ordered_col_idx_buf, partition_size};
      switch (order_col_ti.get_size()) {
        case 8:
          null_range = null_range_info.find_null_range_int<int64_t>(
              column_chunk_owned_.order_columns_.front());
          break;
        case 4:
          null_range = null_range_info.find_null_range_int<int32_t>(
              column_chunk_owned_.order_columns_.front());
          break;
        case 2:
          null_range = null_range_info.find_null_range_int<int16_t>(
              column_chunk_owned_.order_columns_.front());
          break;
        case 1:
          null_range = null_range_info.find_null_range_int<int8_t>(
              column_chunk_owned_.order_columns_.front());
          break;
        default:
          LOG(FATAL) << "Invalid type size: " << order_col_ti.get_size();
      }
    } else if (order_col_ti.is_fp()) {
      const auto null_bit_pattern =
          null_val_bit_pattern(order_col_ti, order_col_ti.get_type() == kFLOAT);
      FindNullRange const null_range_info{
          original_col_idx_buf, ordered_col_idx_buf, partition_size, null_bit_pattern};
      switch (order_col_ti.get_type()) {
        case kFLOAT:
          null_range = null_range_info.find_null_range_fp<float>(
              column_chunk_owned_.order_columns_.front());
          break;
        case kDOUBLE:
          null_range = null_range_info.find_null_range_fp<double>(
              column_chunk_owned_.order_columns_.front());
          break;
        default:
          LOG(FATAL) << "Invalid float type";
      }
    } else {
      LOG(FATAL) << "Invalid column type for window aggregation over the frame";
    }
  }
  cpu_exec_ctx_.ordered_partition_null_start_pos_[partition_idx] = null_range.first;
  cpu_exec_ctx_.ordered_partition_null_end_pos_[partition_idx] = null_range.second + 1;
}

std::vector<WindowFunctionContext::Comparator> WindowFunctionContext::createComparator(
    size_t partition_idx) {
  // create tuple comparator
  std::vector<WindowFunctionContext::Comparator> partition_comparator;
  const auto& order_keys = window_func_->getOrderKeys();
  const auto& collation = window_func_->getCollation();
  CHECK_EQ(order_keys.size(), collation.size());
  for (size_t order_column_idx = 0;
       order_column_idx < column_chunk_owned_.order_columns_.size();
       ++order_column_idx) {
    auto order_column_buffer = column_chunk_owned_.order_columns_[order_column_idx];
    const auto order_col =
        dynamic_cast<const Analyzer::ColumnVar*>(order_keys[order_column_idx].get());
    CHECK(order_col);
    const auto& order_col_collation = collation[order_column_idx];
    auto comparator =
        makeComparator(order_col,
                       order_column_buffer,
                       getPayloadBuf(ExecutorDeviceType::CPU) +
                           getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx],
                       !order_col_collation.is_desc,
                       order_col_collation.nulls_first);
    if (order_col_collation.is_desc) {
      comparator = [comparator](const int64_t lhs, const int64_t rhs) {
        return comparator(rhs, lhs);
      };
    }
    partition_comparator.push_back(comparator);
  }
  return partition_comparator;
}

void WindowFunctionContext::sortPartition(const size_t partition_idx,
                                          int64_t* output_for_partition_buff) {
  const size_t partition_size{
      static_cast<size_t>(getCountBuf(ExecutorDeviceType::CPU)[partition_idx])};
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
    std::sort(output_for_partition_buff,
              output_for_partition_buff + partition_size,
              col_tuple_comparator);
  }
}

const Analyzer::WindowFunction* WindowFunctionContext::getWindowFunction() const {
  return window_func_;
}

const int8_t* WindowFunctionContext::output() const {
  if (device_type_ == ExecutorDeviceType::CPU) {
    return cpu_exec_ctx_.output_;
  } else {
    return gpu_exec_ctx_.output_gpu_->getMemoryPtr();
  }
}

const int64_t* WindowFunctionContext::sortedPartition() const {
  if (device_type_ == ExecutorDeviceType::CPU) {
    return reinterpret_cast<const int64_t*>(cpu_exec_ctx_.sorted_partition_buf_);
  } else {
    return reinterpret_cast<const int64_t*>(
        gpu_exec_ctx_.sorted_partition_buf_gpu_->getMemoryPtr());
  }
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
  if (device_type_ == ExecutorDeviceType::CPU) {
    return cpu_exec_ctx_.partition_start_offset_.data();
  } else {
    CHECK(gpu_exec_ctx_.partition_start_offset_gpu_);
    return reinterpret_cast<int64_t*>(
        gpu_exec_ctx_.partition_start_offset_gpu_->getMemoryPtr());
  }
}

const int64_t* WindowFunctionContext::partitionNumCountBuf() const {
  return partitionStartOffset() + 1;
}

int64_t WindowFunctionContext::aggregateStatePendingOutputs() const {
  CHECK(window_function_is_aggregate(window_func_->getKind()));
  return reinterpret_cast<int64_t>(&aggregate_state_.outputs);
}

const int8_t* WindowFunctionContext::getPartitionStartBitmapBuf(
    ExecutorDeviceType const dt) const {
  if (elem_count_ == 0) {
    return nullptr;
  }
  if (dt == ExecutorDeviceType::CPU) {
    return cpu_exec_ctx_.partition_start_;
  } else {
    CHECK_EQ(dt, ExecutorDeviceType::GPU);
    return gpu_exec_ctx_.partition_start_gpu_->getMemoryPtr();
  }
}

const int8_t* WindowFunctionContext::getPartitionEndBitmapBuf(
    ExecutorDeviceType const dt) const {
  if (elem_count_ == 0) {
    return nullptr;
  }
  if (dt == ExecutorDeviceType::CPU) {
    return cpu_exec_ctx_.partition_end_;
  } else {
    CHECK_EQ(dt, ExecutorDeviceType::GPU);
    return gpu_exec_ctx_.partition_end_gpu_->getMemoryPtr();
  }
}

void WindowFunctionContext::setPartitionStartBitmapSize(size_t sz) {
  cpu_exec_ctx_.partition_start_sz_ = sz;
}

size_t WindowFunctionContext::partitionStartBitmapSize() const {
  return cpu_exec_ctx_.partition_start_sz_;
}

void WindowFunctionContext::setPartitionEndBitmapSize(size_t sz) {
  cpu_exec_ctx_.partition_end_sz_ = sz;
}

size_t WindowFunctionContext::elementCount() const {
  return elem_count_;
}

namespace {

template <class T>
WindowFunctionContext::WindowComparatorResult integer_comparator_asc(
    const int8_t* order_column_buffer,
    const SQLTypeInfo& ti,
    const int32_t* partition_indices,
    const int64_t lhs,
    const int64_t rhs,
    const bool asc_ordering,
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

template <class T>
WindowFunctionContext::WindowComparatorResult integer_comparator_desc(
    const int8_t* order_column_buffer,
    const SQLTypeInfo& ti,
    const int32_t* partition_indices,
    const int64_t lhs,
    const int64_t rhs,
    const bool asc_ordering,
    const bool nulls_first) {
  const auto values = reinterpret_cast<const T*>(order_column_buffer);
  const auto lhs_val = values[partition_indices[lhs]];
  const auto rhs_val = values[partition_indices[rhs]];
  const auto null_val = inline_fixed_encoding_null_val(ti);
  if (lhs_val == null_val && rhs_val == null_val) {
    return WindowFunctionContext::WindowComparatorResult::EQ;
  }
  if (lhs_val == null_val && rhs_val != null_val) {
    return !nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
                        : WindowFunctionContext::WindowComparatorResult::GT;
  }
  if (rhs_val == null_val && lhs_val != null_val) {
    return nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
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
WindowFunctionContext::WindowComparatorResult fp_comparator_asc(
    const int8_t* order_column_buffer,
    const SQLTypeInfo& ti,
    const int32_t* partition_indices,
    const int64_t lhs,
    const int64_t rhs,
    const bool asc_ordering,
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

template <class T, class NullPatternType>
WindowFunctionContext::WindowComparatorResult fp_comparator_desc(
    const int8_t* order_column_buffer,
    const SQLTypeInfo& ti,
    const int32_t* partition_indices,
    const int64_t lhs,
    const int64_t rhs,
    const bool asc_ordering,
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
    return !nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
                        : WindowFunctionContext::WindowComparatorResult::GT;
  }
  if (rhs_bit_pattern == null_bit_pattern && lhs_bit_pattern != null_bit_pattern) {
    return nulls_first ? WindowFunctionContext::WindowComparatorResult::LT
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
    const bool asc_ordering,
    const bool nulls_first) {
  const auto& ti = col_var->get_type_info();
  if (ti.is_integer() || ti.is_decimal() || ti.is_time() || ti.is_boolean()) {
    switch (ti.get_size()) {
      case 8: {
        return [order_column_buffer, nulls_first, partition_indices, asc_ordering, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return asc_ordering ? integer_comparator_asc<int64_t>(order_column_buffer,
                                                                ti,
                                                                partition_indices,
                                                                lhs,
                                                                rhs,
                                                                asc_ordering,
                                                                nulls_first)
                              : integer_comparator_desc<int64_t>(order_column_buffer,
                                                                 ti,
                                                                 partition_indices,
                                                                 lhs,
                                                                 rhs,
                                                                 asc_ordering,
                                                                 nulls_first);
        };
      }
      case 4: {
        return [order_column_buffer, nulls_first, partition_indices, asc_ordering, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return asc_ordering ? integer_comparator_asc<int32_t>(order_column_buffer,
                                                                ti,
                                                                partition_indices,
                                                                lhs,
                                                                rhs,
                                                                asc_ordering,
                                                                nulls_first)
                              : integer_comparator_desc<int32_t>(order_column_buffer,
                                                                 ti,
                                                                 partition_indices,
                                                                 lhs,
                                                                 rhs,
                                                                 asc_ordering,
                                                                 nulls_first);
        };
      }
      case 2: {
        return [order_column_buffer, nulls_first, partition_indices, asc_ordering, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return asc_ordering ? integer_comparator_asc<int16_t>(order_column_buffer,
                                                                ti,
                                                                partition_indices,
                                                                lhs,
                                                                rhs,
                                                                asc_ordering,
                                                                nulls_first)
                              : integer_comparator_desc<int16_t>(order_column_buffer,
                                                                 ti,
                                                                 partition_indices,
                                                                 lhs,
                                                                 rhs,
                                                                 asc_ordering,
                                                                 nulls_first);
        };
      }
      case 1: {
        return [order_column_buffer, nulls_first, partition_indices, asc_ordering, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return asc_ordering ? integer_comparator_asc<int8_t>(order_column_buffer,
                                                               ti,
                                                               partition_indices,
                                                               lhs,
                                                               rhs,
                                                               asc_ordering,
                                                               nulls_first)
                              : integer_comparator_desc<int8_t>(order_column_buffer,
                                                                ti,
                                                                partition_indices,
                                                                lhs,
                                                                rhs,
                                                                asc_ordering,
                                                                nulls_first);
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
        return [order_column_buffer, nulls_first, partition_indices, asc_ordering, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return asc_ordering ? fp_comparator_asc<float, int32_t>(order_column_buffer,
                                                                  ti,
                                                                  partition_indices,
                                                                  lhs,
                                                                  rhs,
                                                                  asc_ordering,
                                                                  nulls_first)
                              : fp_comparator_desc<float, int32_t>(order_column_buffer,
                                                                   ti,
                                                                   partition_indices,
                                                                   lhs,
                                                                   rhs,
                                                                   asc_ordering,
                                                                   nulls_first);
        };
      }
      case kDOUBLE: {
        return [order_column_buffer, nulls_first, partition_indices, asc_ordering, &ti](
                   const int64_t lhs, const int64_t rhs) {
          return asc_ordering ? fp_comparator_asc<double, int64_t>(order_column_buffer,
                                                                   ti,
                                                                   partition_indices,
                                                                   lhs,
                                                                   rhs,
                                                                   asc_ordering,
                                                                   nulls_first)
                              : fp_comparator_desc<double, int64_t>(order_column_buffer,
                                                                    ti,
                                                                    partition_indices,
                                                                    lhs,
                                                                    rhs,
                                                                    asc_ordering,
                                                                    nulls_first);
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
  const size_t partition_size{
      static_cast<size_t>(getCountBuf(ExecutorDeviceType::CPU)[partition_idx])};
  if (partition_size == 0) {
    return;
  }
  const auto offset = getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx];
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
      const auto partition_row_offsets = getPayloadBuf(ExecutorDeviceType::CPU) + offset;
      apply_lag_to_partition(
          lag_or_lead, partition_row_offsets, output_for_partition_buff, partition_size);
      break;
    }
    case SqlWindowFunctionKind::FIRST_VALUE:
    case SqlWindowFunctionKind::LAST_VALUE: {
      const auto target_idx =
          get_target_idx_for_first_or_last_value_func(window_func, partition_size);
      const auto partition_row_offsets = getPayloadBuf(ExecutorDeviceType::CPU) + offset;
      apply_nth_value_to_partition(
          partition_row_offsets, output_for_partition_buff, partition_size, target_idx);
      break;
    }
    case SqlWindowFunctionKind::NTH_VALUE: {
      auto const n_value_ptr =
          dynamic_cast<Analyzer::Constant*>(window_func_->getArgs()[1].get());
      CHECK(n_value_ptr);
      auto const n_value = static_cast<size_t>(n_value_ptr->get_constval().intval);
      const auto partition_row_offsets = getPayloadBuf(ExecutorDeviceType::CPU) + offset;
      if (n_value < partition_size) {
        apply_nth_value_to_partition(
            partition_row_offsets, output_for_partition_buff, partition_size, n_value);
      } else {
        // when NTH_VALUE of the current row is NULL, we keep the NULL value in the
        // current row's output storage in the query output buffer, so we assign the
        // original index of the current row to the corresponding slot in
        // `output_for_partition_buff`
        apply_original_index_to_partition(
            partition_row_offsets, output_for_partition_buff, partition_size);
      }
      break;
    }
    case SqlWindowFunctionKind::AVG:
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX:
    case SqlWindowFunctionKind::SUM:
    case SqlWindowFunctionKind::COUNT:
    case SqlWindowFunctionKind::CONDITIONAL_CHANGE_EVENT:
    case SqlWindowFunctionKind::LEAD_IN_FRAME:
    case SqlWindowFunctionKind::LAG_IN_FRAME:
    case SqlWindowFunctionKind::NTH_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::FIRST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::LAST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::COUNT_IF:
    case SqlWindowFunctionKind::SUM_IF:
    case SqlWindowFunctionKind::FORWARD_FILL:
    case SqlWindowFunctionKind::BACKWARD_FILL: {
      if (window_function_requires_peer_handling(window_func)) {
        index_to_partition_end(getPartitionEndBitmapBuf(ExecutorDeviceType::CPU),
                               offset,
                               output_for_partition_buff,
                               partition_size,
                               col_tuple_comparator);
      }
      const auto partition_row_offsets = getPayloadBuf(ExecutorDeviceType::CPU) + offset;
      apply_permutation_to_partition(
          output_for_partition_buff, partition_row_offsets, partition_size);
      break;
    }
    default: {
      std::ostringstream oss;
      oss << "Window function not supported yet: " << window_func_->getKind();
      throw std::runtime_error(oss.str());
    }
  }
}

void WindowFunctionContext::resizeStorageForWindowFraming(bool for_reuse) {
  auto const partition_count = getNumWindowPartition();
  aggregate_trees_.resizeStorageForWindowFraming(partition_count);
  if (!for_reuse) {
    aggregate_trees_.segment_trees_owned_.resize(partition_count);
  }
}

namespace {
bool allow_framing_on_time_or_date(SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::CONDITIONAL_CHANGE_EVENT:
      return true;
    default:
      return false;
  }
}
}  // namespace

void WindowFunctionContext::buildAggregationTreeForPartition(
    SqlWindowFunctionKind agg_type,
    size_t partition_idx,
    size_t partition_size,
    const int32_t* original_rowid_buf,
    const int64_t* ordered_rowid_buf,
    const SQLTypeInfo& input_col_ti) {
  if (!(input_col_ti.is_number() || input_col_ti.is_boolean() ||
        input_col_ti.is_time_or_date())) {
    throw QueryNotSupported("Window aggregate function over frame on a column type " +
                            ::toString(input_col_ti.get_type()) + " is not supported.");
  }
  if (input_col_ti.is_time_or_date() &&
      !allow_framing_on_time_or_date(window_func_->getKind()) &&
      !(agg_type == SqlWindowFunctionKind::MIN ||
        agg_type == SqlWindowFunctionKind::MAX ||
        agg_type == SqlWindowFunctionKind::COUNT)) {
    throw QueryNotSupported(
        "Aggregation over a window frame for a column type " +
        ::toString(input_col_ti.get_type()) +
        " must use one of the following window aggregate function: MIN / MAX / COUNT");
  }
  const auto type = input_col_ti.is_decimal() ? decimal_to_int_type(input_col_ti)
                    : input_col_ti.is_time_or_date()
                        ? get_int_type_by_size(input_col_ti.get_size())
                        : input_col_ti.get_type();
  if (partition_size > 0) {
    const int64_t* ordered_rowid_buf_for_partition =
        ordered_rowid_buf + getOffsetBuf(ExecutorDeviceType::CPU)[partition_idx];
    switch (type) {
      case kBOOLEAN:
      case kTINYINT: {
        auto segment_tree = std::make_unique<SegmentTree<int8_t, int64_t>>(
            column_chunk_owned_.window_func_expr_columns_,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            partition_size,
            aggregate_trees_.aggregate_tree_infos_[partition_idx],
            aggregate_trees_.aggregate_tree_sizes_in_byte_[partition_idx],
            aggregate_trees_.aggregate_tree_start_offset_[partition_idx],
            aggregate_trees_.aggregate_tree_holder_,
            agg_type,
            aggregate_trees_fan_out_);
        aggregate_trees_.aggregate_trees_depth_[partition_idx] =
            segment_tree ? segment_tree->getLeafDepth() : 0;
        aggregate_trees_.aggregate_trees_leaf_start_offset_[partition_idx] =
            segment_tree ? segment_tree->computeLeafStartOffset() : 0;
        aggregate_trees_.segment_trees_owned_[partition_idx] = std::move(segment_tree);
        break;
      }
      case kSMALLINT: {
        auto segment_tree = std::make_unique<SegmentTree<int16_t, int64_t>>(
            column_chunk_owned_.window_func_expr_columns_,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            partition_size,
            aggregate_trees_.aggregate_tree_infos_[partition_idx],
            aggregate_trees_.aggregate_tree_sizes_in_byte_[partition_idx],
            aggregate_trees_.aggregate_tree_start_offset_[partition_idx],
            aggregate_trees_.aggregate_tree_holder_,
            agg_type,
            aggregate_trees_fan_out_);
        aggregate_trees_.aggregate_trees_depth_[partition_idx] =
            segment_tree ? segment_tree->getLeafDepth() : 0;
        aggregate_trees_.aggregate_trees_leaf_start_offset_[partition_idx] =
            segment_tree ? segment_tree->computeLeafStartOffset() : 0;
        aggregate_trees_.segment_trees_owned_[partition_idx] = std::move(segment_tree);
        break;
      }
      case kINT: {
        auto segment_tree = std::make_unique<SegmentTree<int32_t, int64_t>>(
            column_chunk_owned_.window_func_expr_columns_,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            partition_size,
            aggregate_trees_.aggregate_tree_infos_[partition_idx],
            aggregate_trees_.aggregate_tree_sizes_in_byte_[partition_idx],
            aggregate_trees_.aggregate_tree_start_offset_[partition_idx],
            aggregate_trees_.aggregate_tree_holder_,
            agg_type,
            aggregate_trees_fan_out_);
        aggregate_trees_.aggregate_trees_depth_[partition_idx] =
            segment_tree ? segment_tree->getLeafDepth() : 0;
        aggregate_trees_.aggregate_trees_leaf_start_offset_[partition_idx] =
            segment_tree ? segment_tree->computeLeafStartOffset() : 0;
        aggregate_trees_.segment_trees_owned_[partition_idx] = std::move(segment_tree);
        break;
      }
      case kDECIMAL:
      case kNUMERIC:
      case kBIGINT: {
        auto segment_tree = std::make_unique<SegmentTree<int64_t, int64_t>>(
            column_chunk_owned_.window_func_expr_columns_,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            partition_size,
            aggregate_trees_.aggregate_tree_infos_[partition_idx],
            aggregate_trees_.aggregate_tree_sizes_in_byte_[partition_idx],
            aggregate_trees_.aggregate_tree_start_offset_[partition_idx],
            aggregate_trees_.aggregate_tree_holder_,
            agg_type,
            aggregate_trees_fan_out_);
        aggregate_trees_.aggregate_trees_depth_[partition_idx] =
            segment_tree ? segment_tree->getLeafDepth() : 0;
        aggregate_trees_.aggregate_trees_leaf_start_offset_[partition_idx] =
            segment_tree ? segment_tree->computeLeafStartOffset() : 0;
        aggregate_trees_.segment_trees_owned_[partition_idx] = std::move(segment_tree);
        break;
      }
      case kFLOAT: {
        auto segment_tree = std::make_unique<SegmentTree<float, double>>(
            column_chunk_owned_.window_func_expr_columns_,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            partition_size,
            aggregate_trees_.aggregate_tree_infos_[partition_idx],
            aggregate_trees_.aggregate_tree_sizes_in_byte_[partition_idx],
            aggregate_trees_.aggregate_tree_start_offset_[partition_idx],
            aggregate_trees_.aggregate_tree_holder_,
            agg_type,
            aggregate_trees_fan_out_);
        aggregate_trees_.aggregate_trees_depth_[partition_idx] =
            segment_tree ? segment_tree->getLeafDepth() : 0;
        aggregate_trees_.aggregate_trees_leaf_start_offset_[partition_idx] =
            segment_tree ? segment_tree->computeLeafStartOffset() : 0;
        aggregate_trees_.segment_trees_owned_[partition_idx] = std::move(segment_tree);
        break;
      }
      case kDOUBLE: {
        auto segment_tree = std::make_unique<SegmentTree<double, double>>(
            column_chunk_owned_.window_func_expr_columns_,
            input_col_ti,
            original_rowid_buf,
            ordered_rowid_buf_for_partition,
            partition_size,
            aggregate_trees_.aggregate_tree_infos_[partition_idx],
            aggregate_trees_.aggregate_tree_sizes_in_byte_[partition_idx],
            aggregate_trees_.aggregate_tree_start_offset_[partition_idx],
            aggregate_trees_.aggregate_tree_holder_,
            agg_type,
            aggregate_trees_fan_out_);
        aggregate_trees_.aggregate_trees_depth_[partition_idx] =
            segment_tree ? segment_tree->getLeafDepth() : 0;
        aggregate_trees_.aggregate_trees_leaf_start_offset_[partition_idx] =
            segment_tree ? segment_tree->computeLeafStartOffset() : 0;
        aggregate_trees_.segment_trees_owned_[partition_idx] = std::move(segment_tree);
        break;
      }
      default:
        UNREACHABLE() << type;
    }
  } else {
    // handling a case of an empty partition
    aggregate_trees_.aggregate_trees_depth_[partition_idx] = 0;
    aggregate_trees_.aggregate_trees_leaf_start_offset_[partition_idx] = 0;
    if (input_col_ti.is_integer() || input_col_ti.is_decimal() ||
        input_col_ti.is_boolean() || input_col_ti.is_time_or_date()) {
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_integer_type_[partition_idx] =
            nullptr;
      } else {
        aggregate_trees_.aggregate_tree_for_integer_type_[partition_idx] = nullptr;
      }
    } else {
      CHECK(input_col_ti.is_fp());
      if (agg_type == SqlWindowFunctionKind::AVG) {
        aggregate_trees_.derived_aggregate_tree_for_double_type_[partition_idx] = nullptr;
      } else {
        aggregate_trees_.aggregate_tree_for_double_type_[partition_idx] = nullptr;
      }
    }
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

const size_t* WindowFunctionContext::getAggregateTreeDepth() const {
  if (device_type_ == ExecutorDeviceType::CPU) {
    return aggregate_trees_.aggregate_trees_depth_.data();
  } else {
    return reinterpret_cast<size_t*>(
        aggregate_trees_.aggregate_trees_depth_gpu_->getMemoryPtr());
  }
}

const size_t* WindowFunctionContext::getAggregateTreeLeafStartOffset() const {
  if (device_type_ == ExecutorDeviceType::CPU) {
    return aggregate_trees_.aggregate_trees_leaf_start_offset_.data();
  } else {
    return reinterpret_cast<size_t*>(
        aggregate_trees_.aggregate_trees_leaf_start_offset_gpu_->getMemoryPtr());
  }
}

size_t WindowFunctionContext::getAggregateTreeFanout() const {
  return aggregate_trees_fan_out_;
}

const int64_t* WindowFunctionContext::getNullValueStartPos() const {
  if (device_type_ == ExecutorDeviceType::CPU) {
    return cpu_exec_ctx_.ordered_partition_null_start_pos_.data();
  } else {
    return reinterpret_cast<int64_t*>(
        gpu_exec_ctx_.ordered_partition_null_start_pos_gpu_->getMemoryPtr());
  }
}

const int64_t* WindowFunctionContext::getNullValueEndPos() const {
  if (device_type_ == ExecutorDeviceType::CPU) {
    return cpu_exec_ctx_.ordered_partition_null_end_pos_.data();
  } else {
    return reinterpret_cast<int64_t*>(
        gpu_exec_ctx_.ordered_partition_null_end_pos_gpu_->getMemoryPtr());
  }
}

void WindowFunctionContext::fillPartitionStart() {
  CountDistinctDescriptor partition_start_bitmap{CountDistinctImplType::Bitmap,
                                                 0,
                                                 0,
                                                 static_cast<int64_t>(elem_count_),
                                                 false,
                                                 ExecutorDeviceType::CPU,
                                                 1};
  auto bitmap_sz = partition_start_bitmap.bitmapPaddedSizeBytes();
  if (partitions_) {
    bitmap_sz += partitions_->isBitwiseEq() ? 1 : 0;
  }
  setPartitionStartBitmapSize(bitmap_sz);
  cpu_exec_ctx_.partition_start_ = static_cast<int8_t*>(checked_calloc(bitmap_sz, 1));
  int64_t partition_count = getNumWindowPartition();
  auto partition_start_handle = reinterpret_cast<int64_t>(cpu_exec_ctx_.partition_start_);
  agg_count_distinct_bitmap(&partition_start_handle, 0, 0, 0);
  if (!cpu_exec_ctx_.partition_start_offset_.empty()) {
    // if we have `partition_start_offset_`, we can reuse it for this logic
    // but note that it has partition_count + 1 elements where the first element is zero
    // which means the first partition's start offset is zero
    // and rest of them can represent values required for this logic
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      agg_count_distinct_bitmap(
          &partition_start_handle, cpu_exec_ctx_.partition_start_offset_[i + 1], 0, 0);
    }
  } else {
    std::vector<size_t> partition_offsets(partition_count);
    std::partial_sum(getCountBuf(ExecutorDeviceType::CPU),
                     getCountBuf(ExecutorDeviceType::CPU) + partition_count,
                     partition_offsets.begin());
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      agg_count_distinct_bitmap(&partition_start_handle, partition_offsets[i], 0, 0);
    }
  }
}

void WindowFunctionContext::fillPartitionEnd() {
  CountDistinctDescriptor partition_start_bitmap{CountDistinctImplType::Bitmap,
                                                 0,
                                                 0,
                                                 static_cast<int64_t>(elem_count_),
                                                 false,
                                                 ExecutorDeviceType::CPU,
                                                 1};
  auto bitmap_sz = partition_start_bitmap.bitmapPaddedSizeBytes();
  if (partitions_) {
    bitmap_sz += partitions_->isBitwiseEq() ? 1 : 0;
  }
  setPartitionEndBitmapSize(bitmap_sz);
  cpu_exec_ctx_.partition_end_ = static_cast<int8_t*>(checked_calloc(bitmap_sz, 1));
  auto partition_end_handle = reinterpret_cast<int64_t>(cpu_exec_ctx_.partition_end_);
  int64_t partition_count = getNumWindowPartition();
  if (!cpu_exec_ctx_.partition_start_offset_.empty()) {
    // if we have `partition_start_offset_`, we can reuse it for this logic
    // but note that it has partition_count + 1 elements where the first element is zero
    // which means the first partition's start offset is zero
    // and rest of them can represent values required for this logic
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      if (cpu_exec_ctx_.partition_start_offset_[i + 1] == 0) {
        continue;
      }
      agg_count_distinct_bitmap(
          &partition_end_handle, cpu_exec_ctx_.partition_start_offset_[i + 1] - 1, 0, 0);
    }
    if (elem_count_) {
      agg_count_distinct_bitmap(&partition_end_handle, elem_count_ - 1, 0, 0);
    }
  } else {
    std::vector<size_t> partition_offsets(partition_count);
    std::partial_sum(getCountBuf(ExecutorDeviceType::CPU),
                     getCountBuf(ExecutorDeviceType::CPU) + partition_count,
                     partition_offsets.begin());
    for (int64_t i = 0; i < partition_count - 1; ++i) {
      if (partition_offsets[i] == 0) {
        continue;
      }
      agg_count_distinct_bitmap(&partition_end_handle, partition_offsets[i] - 1, 0, 0);
    }
    if (elem_count_) {
      agg_count_distinct_bitmap(&partition_end_handle, elem_count_ - 1, 0, 0);
    }
  }
}

const int8_t* WindowFunctionContext::getPartitionBuf(ExecutorDeviceType const dt) const {
  CHECK(partitions_);
  return partitions_->getJoinHashBuffer(dt, device_id_);
}

const int32_t* WindowFunctionContext::getPayloadBuf(ExecutorDeviceType const dt) const {
  if (partitions_) {
    return reinterpret_cast<const int32_t*>(getPartitionBuf(dt) +
                                            partitions_->payloadBufferOff());
  }
  if (dt == ExecutorDeviceType::CPU) {
    return cpu_exec_ctx_.dummy_payload_;
  } else {
    CHECK_EQ(dt, ExecutorDeviceType::GPU);
    return reinterpret_cast<int32_t*>(gpu_exec_ctx_.dummy_payload_gpu_->getMemoryPtr());
  }
}

const int32_t* WindowFunctionContext::getOffsetBuf(ExecutorDeviceType const dt) const {
  if (partitions_) {
    return reinterpret_cast<const int32_t*>(getPartitionBuf(dt) +
                                            partitions_->offsetBufferOff());
  }
  if (dt == ExecutorDeviceType::CPU) {
    return &cpu_exec_ctx_.dummy_offset_;
  } else {
    CHECK_EQ(dt, ExecutorDeviceType::GPU);
    return reinterpret_cast<int32_t*>(gpu_exec_ctx_.dummy_offset_gpu_->getMemoryPtr());
  }
}

const int32_t* WindowFunctionContext::getCountBuf(ExecutorDeviceType const dt) const {
  if (partitions_) {
    return reinterpret_cast<const int32_t*>(getPartitionBuf(dt) +
                                            partitions_->countBufferOff());
  }
  if (dt == ExecutorDeviceType::CPU) {
    return &cpu_exec_ctx_.dummy_count_;
  } else {
    CHECK_EQ(dt, ExecutorDeviceType::GPU);
    return reinterpret_cast<int32_t*>(gpu_exec_ctx_.dummy_count_gpu_->getMemoryPtr());
  }
}

size_t WindowFunctionContext::getNumWindowPartition() const {
  if (partitions_) {
    auto constexpr cpu_dt = ExecutorDeviceType::CPU;
    const auto partition_count = getCountBuf(cpu_dt) - getOffsetBuf(cpu_dt);
    CHECK_GE(partition_count, 0);
    return partition_count;
  }
  return 1;  // non-partitioned window function
}

const bool WindowFunctionContext::needsToBuildAggregateTree() const {
  return window_func_->hasFraming() &&
         window_func_->hasAggregateTreeRequiredWindowFunc() && elem_count_ > 0;
}

QueryPlanHash const WindowFunctionContext::computeAggregateTreeCacheKey() const {
  // aggregate tree is constructed per window aggregate function kind, input expression,
  // partition key(s) and ordering key
  //  this means when two window definitions have the same condition listed above but
  //  differ in frame bound declaration,
  // they can share the same aggregate tree
  auto cache_key = boost::hash_value(::toString(window_func_->getKind()));
  boost::hash_combine(cache_key, ::toString(window_func_->getArgs()));
  boost::hash_combine(cache_key, ::toString(window_func_->getPartitionKeys()));
  boost::hash_combine(cache_key, ::toString(window_func_->getOrderKeys()));
  for (auto& order_entry : window_func_->getCollation()) {
    boost::hash_combine(cache_key, order_entry.toString());
  }
  return cache_key;
}

void WindowFunctionContext::prepareEmptyPartition(bool for_gpu) {
  cpu_exec_ctx_.dummy_count_ = elem_count_;
  cpu_exec_ctx_.dummy_offset_ = 0;
  cpu_exec_ctx_.dummy_payload_ =
      reinterpret_cast<int32_t*>(checked_malloc(elem_count_ * sizeof(int32_t)));
  // todo (yoonmin): parallelize this (std::iota)
  std::iota(cpu_exec_ctx_.dummy_payload_,
            cpu_exec_ctx_.dummy_payload_ + elem_count_,
            int32_t(0));
  if (for_gpu) {
    gpu_exec_ctx_.dummy_payload_gpu_ = gpu_exec_ctx_.data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, elem_count_ * sizeof(int32_t));
    device_allocator_->copyToDevice(gpu_exec_ctx_.dummy_payload_gpu_->getMemoryPtr(),
                                    cpu_exec_ctx_.dummy_payload_,
                                    sizeof(int32_t) * elem_count_,
                                    "WindowFunc_DummyPayload");
    gpu_exec_ctx_.dummy_count_gpu_ = gpu_exec_ctx_.data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, sizeof(int32_t));
    device_allocator_->copyToDevice(gpu_exec_ctx_.dummy_count_gpu_->getMemoryPtr(),
                                    &cpu_exec_ctx_.dummy_count_,
                                    sizeof(int32_t),
                                    "WindowFunc_DummyCount");
    gpu_exec_ctx_.dummy_offset_gpu_ = gpu_exec_ctx_.data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, sizeof(int32_t));
    device_allocator_->copyToDevice(gpu_exec_ctx_.dummy_offset_gpu_->getMemoryPtr(),
                                    &cpu_exec_ctx_.dummy_offset_,
                                    sizeof(int32_t),
                                    "WindowFunc_DummyOffset");
  }
}

void WindowFunctionContext::preparePartitionStartOffsetBufs(bool for_gpu,
                                                            size_t partition_count) {
  cpu_exec_ctx_.partition_start_offset_.resize(partition_count + 1);
  // the first partition starts at zero position
  std::exclusive_scan(getCountBuf(ExecutorDeviceType::CPU),
                      getCountBuf(ExecutorDeviceType::CPU) + partition_count,
                      cpu_exec_ctx_.partition_start_offset_.begin(),
                      static_cast<size_t>(0));
  if (for_gpu) {
    gpu_exec_ctx_.partition_start_offset_gpu_ = gpu_exec_ctx_.data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, (partition_count + 1) * sizeof(int64_t));
    device_allocator_->copyToDevice(
        gpu_exec_ctx_.partition_start_offset_gpu_->getMemoryPtr(),
        cpu_exec_ctx_.partition_start_offset_.data(),
        (partition_count + 1) * sizeof(int64_t),
        "WindowFunc_PartitionStartOffset");
  }
}

void WindowFunctionContext::prepareAggregateTreeBufs(bool for_gpu,
                                                     size_t const partition_count) {
  aggregate_trees_.aggregate_trees_depth_.resize(partition_count);
  aggregate_trees_.aggregate_trees_leaf_start_offset_.resize(partition_count);
  cpu_exec_ctx_.ordered_partition_null_start_pos_.resize(partition_count);
  cpu_exec_ctx_.ordered_partition_null_end_pos_.resize(partition_count);
  if (for_gpu) {
    aggregate_trees_.aggregate_trees_depth_gpu_ = gpu_exec_ctx_.data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, partition_count * sizeof(size_t));
    aggregate_trees_.aggregate_trees_leaf_start_offset_gpu_ =
        gpu_exec_ctx_.data_mgr_->alloc(
            MemoryLevel::GPU_LEVEL, device_id_, partition_count * sizeof(size_t));
    gpu_exec_ctx_.ordered_partition_null_start_pos_gpu_ = gpu_exec_ctx_.data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, partition_count * sizeof(size_t));
    gpu_exec_ctx_.ordered_partition_null_end_pos_gpu_ = gpu_exec_ctx_.data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, partition_count * sizeof(size_t));
    aggregate_trees_.resizeStorageForWindowFraming(partition_count);

    aggregate_trees_.aggregate_tree_for_integer_type_buf_gpu_ = data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, partition_count * sizeof(int64_t*));
    aggregate_trees_.aggregate_tree_for_double_type_buf_gpu_ = data_mgr_->alloc(
        MemoryLevel::GPU_LEVEL, device_id_, partition_count * sizeof(double*));
    aggregate_trees_.derived_aggregate_tree_for_integer_type_buf_gpu_ =
        data_mgr_->alloc(MemoryLevel::GPU_LEVEL,
                         device_id_,
                         partition_count * sizeof(SumAndCountPair<int64_t>*));
    aggregate_trees_.derived_aggregate_tree_for_double_type_buf_gpu_ =
        data_mgr_->alloc(MemoryLevel::GPU_LEVEL,
                         device_id_,
                         partition_count * sizeof(SumAndCountPair<double>*));
    aggregate_trees_.aggregate_tree_for_integer_type_gpu_ = reinterpret_cast<int64_t**>(
        aggregate_trees_.aggregate_tree_for_integer_type_buf_gpu_->getMemoryPtr());
    aggregate_trees_.aggregate_tree_for_double_type_gpu_ = reinterpret_cast<double**>(
        aggregate_trees_.aggregate_tree_for_double_type_buf_gpu_->getMemoryPtr());
    aggregate_trees_.derived_aggregate_tree_for_integer_type_gpu_ =
        reinterpret_cast<SumAndCountPair<int64_t>**>(
            aggregate_trees_.derived_aggregate_tree_for_integer_type_buf_gpu_
                ->getMemoryPtr());
    aggregate_trees_.derived_aggregate_tree_for_double_type_gpu_ =
        reinterpret_cast<SumAndCountPair<double>**>(
            aggregate_trees_.derived_aggregate_tree_for_double_type_buf_gpu_
                ->getMemoryPtr());
  }
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

#ifdef HAVE_CUDA
void WindowProjectNodeContext::registerDeviceAllocator(
    int device_id,
    Data_Namespace::DataMgr* data_mgr) {
  auto [itr, emplaced] = device_allocator_owned_.emplace(
      device_id,
      std::make_unique<CudaAllocator>(
          data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id)));
  CHECK(emplaced);
}

CudaAllocator* WindowProjectNodeContext::getDeviceAllocator(int device_id) const {
  auto it = device_allocator_owned_.find(device_id);
  CHECK(it != device_allocator_owned_.end());
  return it->second.get();
}
#endif

void WindowProjectNodeContext::registerWindowPartition(
    size_t const target_idx,
    std::shared_ptr<HashJoin> partition) {
  window_partition_owned_.emplace(target_idx, std::move(partition));
}

HashJoin* WindowProjectNodeContext::getWindowPartition(size_t const target_idx) const {
  auto it = window_partition_owned_.find(target_idx);
  CHECK(it != window_partition_owned_.end());
  return it->second.get();
}
